import os
import gym
import d4rl
import h5py
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters

from cleandiffuser_ex.utils import set_seed
from cleandiffuser_ex.diffusion import DiscreteDiffusionSDEEX
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexIVFWrapper


@hydra.main(config_path="../configs/lomapdiffuser/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoDataset(
        env.get_dataset(), horizon=args.task.horizon, terminal_penalty=args.terminal_penalty, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDEEX(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        progress_bar = tqdm(total=args.diffusion_gradient_steps, desc="Training Progress")

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)

            x = torch.cat([obs, act], -1)

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(x)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
                classifier_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")

            n_gradient_step += 1
            progress_bar.update(1)

            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    elif args.mode == "prepare_data":

        dataset_size = min(1000000, len(dataset))
        normalizer = dataset.get_normalizer()

        traj_dataset = np.zeros((dataset_size, args.task.horizon, obs_dim + act_dim), dtype=np.float32)

        gen_dl = DataLoader(
            dataset, batch_size=5000, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )

        ptr = 0
        with tqdm(total=dataset_size, desc=f"prepare_data: {ptr}/{dataset_size}", leave=False) as pbar:
            for batch in gen_dl:
                obs = batch["obs"]["state"].to(args.device)
                act = batch["act"].to(args.device)
                x = torch.cat([obs, act], -1)
                bs = x.shape[0]

                if ptr + bs > dataset_size:
                    bs = dataset_size - ptr

                traj_dataset[ptr:ptr+bs] = x[:bs].cpu().numpy()

                ptr += bs

                pbar.update(bs)

                if ptr >= dataset_size:
                    break

        with h5py.File(save_path + "dataset.h5", "w") as f:
            f.create_dataset(f"traj_dataset", data=traj_dataset)

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")

        agent.eval()

        if not args.task.proj_range:
            faiss_wrapper = None
        else:
            with h5py.File(save_path + "dataset.h5", "r") as f:
                traj_dataset = f["traj_dataset"][:]

                dim_weights = np.ones((args.task.horizon, obs_dim + act_dim), dtype=np.float32)
                dim_weights[:, obs_dim:] = args.task.action_dim_weight
                dim_weights[0, :] = args.task.s_weight
                faiss_wrapper = FaissIndexIVFWrapper(
                    similarity_metric=args.faiss_similarity,
                    nlist=args.faiss_nlist,
                    data=traj_dataset,
                    dim_weights=dim_weights,
                    device=args.device)

        env_eval = gym.vector.make(args.task.env_name, args.num_envs, asynchronous=False)
        env_eval.seed(args.seed)
        normalizer = dataset.get_normalizer()

        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        for i in tqdm(range(args.num_episodes), desc="Inference Episodes"):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            with tqdm(total=1001, desc=f"Episode {i+1}/{args.num_episodes}", leave=False) as pbar:
                while not np.all(cum_done) and t < 1000 + 1:
                    # normalize obs
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                    # sample trajectories
                    prior[:, 0, :obs_dim] = obs
                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature,
                        faiss_wrapper=faiss_wrapper, proj_range=args.task.proj_range,
                        n_manifold_samples=args.task.n_manifold_samples, tau=args.tau)

                    # select the best plan
                    logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                    idx = logp.argmax(0)
                    act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                        idx, torch.arange(args.num_envs), 0, obs_dim:]
                    act = act.clip(-1., 1.).cpu().numpy()

                    # step
                    obs, rew, done, info = env_eval.step(act)

                    pbar.update(1)
                    t += 1
                    
                    cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                    ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                    print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}, '
                          f'logp: {logp[idx, torch.arange(args.num_envs)]}')
                    
            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).reshape(args.num_episodes * args.num_envs)
        episode_rewards *= 100

        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        result_str = f"scores: {mean:.1f} +/- {err:.2f}"
        print(result_str)

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()