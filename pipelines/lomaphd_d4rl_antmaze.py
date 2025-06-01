import os
import gym
import d4rl
import h5py
import hydra
import einops
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import MultiHorizonD4RLAntmazeDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.invdynamic import FancyMlpInvDynamic

from cleandiffuser_ex.utils import set_seed
from cleandiffuser_ex.diffusion import DiscreteDiffusionSDEEX
from cleandiffuser_ex.faiss_index_wrapper import FaissIndexIVFWrapper


@hydra.main(config_path="../configs/lomaphd/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    planning_horizons = [16, 16]
    # ========================== Level Setup ==========================
    n_levels = len(planning_horizons)
    temporal_horizons = [planning_horizons[-1] for _ in range(n_levels)]
    for i in range(n_levels - 1):
        temporal_horizons[-2 - i] = (planning_horizons[-2 - i] - 1) * (temporal_horizons[-1 - i] - 1) + 1

    env = gym.make(args.task.env_name)
    dataset = MultiHorizonD4RLAntmazeDataset(
        env.get_dataset(), horizons=temporal_horizons, 
        noreaching_penalty=args.noreaching_penalty, discount=args.discount)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # =========================== Model Setup ==========================
    fix_masks = [torch.zeros((h, obs_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, obs_dim)) for h in planning_horizons]
    for i in range(n_levels):
        fix_idx = 0 if i == 0 else [0, -1]
        fix_masks[i][fix_idx, :] = 1.
        loss_weights[i][1, :] = args.next_obs_loss_weight

    nn_diffusions = [
        JannerUNet1d(
            obs_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
            timestep_emb_type="positional", attention=False, kernel_size=5)
        for _ in range(n_levels)]

    diffusions = [
        DiscreteDiffusionSDEEX(
            nn_diffusions[i], None,
            fix_mask=fix_masks[i], loss_weight=loss_weights[i], classifier=None, ema_rate=args.ema_rate,
            device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)
        for i in range(n_levels)]

    nn_classifier = HalfJannerUNet1d(
        planning_horizons[0], obs_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    diffusions[0].classifier = classifier

    invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, nn.Tanh(), add_dropout=True, device=args.device)

    if args.mode == "train":

        progress_bar = tqdm(total=args.diffusion_gradient_steps, desc="Training Progress")

        diffusion_lr_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(diffusions[i].optimizer, args.diffusion_gradient_steps)
            for i in range(n_levels)]

        classifier_lr_scheduler = CosineAnnealingLR(diffusions[0].classifier.optim, args.classifier_gradient_steps)

        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)

        for diffusion in diffusions:
            diffusion.train()
        invdyn.train()

        n_gradient_step = 0
        log = dict.fromkeys(
            [f"diffusion_loss{i}" for i in range(n_levels)] 
            + ["classifier_loss"] 
            + ["invdyn_loss"], 0.)
        for batch in loop_dataloader(dataloader):
            for i in range(n_levels):

                batch_data = batch[i]["data"]

                obs = batch_data["obs"]["state"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(
                    args.device)
                act = batch_data["act"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(args.device)
                val = batch_data["val"].to(args.device)

                log[f"diffusion_loss{i}"] += diffusions[i].update(obs)["loss"]
                diffusion_lr_schedulers[i].step()

                if i == 0 and n_gradient_step <= args.classifier_gradient_steps:
                    log[f"classifier_loss"] += diffusions[i].update_classifier(obs, val)['loss']
                    classifier_lr_scheduler.step()

                if i == n_levels - 1 and n_gradient_step < args.invdyn_gradient_steps:
                    log[f"invdyn_loss"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])["loss"]
                    invdyn_lr_scheduler.step()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys(
                    [f"diffusion_loss{i}" for i in range(n_levels)] 
                    + ["classifier_loss"] 
                    + ["invdyn_loss"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                for i in range(n_levels):
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_{n_gradient_step + 1}.pt')
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_latest.pt')
                classifier.save(save_path + f'classifier_ckpt_{n_gradient_step + 1}.pt')
                classifier.save(save_path + f'classifier_ckpt_latest.pt')
                invdyn.save(save_path + f'invdyn_ckpt_{n_gradient_step + 1}.pt')
                invdyn.save(save_path + f'invdyn_ckpt_latest.pt')

            n_gradient_step += 1
            progress_bar.update(1)

            if n_gradient_step > args.diffusion_gradient_steps:
                break

    elif args.mode == "prepare_data":

        dataset_size = min(500000, len(dataset))

        traj_dataset = []
        for i in range(n_levels):
            traj_dataset.append(
                np.zeros((dataset_size, planning_horizons[i], obs_dim), dtype=np.float32))

        gen_dl = DataLoader(
            dataset, batch_size=5000, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )

        ptr = 0
        with tqdm(total=dataset_size, desc=f"prepare_data: {ptr}/{dataset_size}", leave=False) as pbar:
            for batch in gen_dl:
                for i in range(n_levels):
                    batch_data = batch[i]["data"]
                    obs = batch_data["obs"]["state"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(
                        args.device)

                    bs = obs.shape[0]

                    traj_dataset[i][ptr:ptr+bs] = obs[:bs].cpu().numpy()

                ptr += bs

                pbar.update(bs)

                if ptr >= dataset_size:
                    break

        with h5py.File(save_path + "dataset.h5", "w") as f:
            for i in range(n_levels):
                f.create_dataset(f"traj_dataset_{i}", data=traj_dataset[i])

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        
        for i in range(n_levels):
            diffusions[i].load(
                save_path + f'{"diffusion"}{i}_ckpt_{args.diffusion_ckpt}.pt')
            if i == 0:
                diffusions[i].classifier.load(save_path + f"classifier_ckpt_{args.classifier_ckpt}.pt")
            diffusions[i].eval()
        invdyn.load(save_path + f'invdyn_ckpt_{args.invdyn_ckpt}.pt')
        invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs, asynchronous=False)
        env_eval.seed(args.seed)
        normalizer = dataset.get_normalizer()

        if not args.task.proj_range:
            faiss_wrapper = None
        else:
            with h5py.File(save_path + "dataset.h5", "r") as f:
                traj_dataset_0 = f["traj_dataset_0"][:]

                dim_weights = np.zeros((planning_horizons[0], obs_dim), dtype=np.float32)
                dim_weights[:, :2] = 1. 
                dim_weights[0, :2] = args.task.s_weight

                faiss_wrapper = FaissIndexIVFWrapper(
                    similarity_metric=args.faiss_similarity,
                    nlist=args.faiss_nlist,
                    data=traj_dataset_0,
                    dim_weights=dim_weights,
                    device=args.device)

        episode_rewards = []
        priors = [torch.zeros((args.num_envs, planning_horizons[i], obs_dim),
                              device=args.device) for i in range(n_levels)]
        priors[0] = torch.zeros((args.num_envs, args.num_candidates, planning_horizons[0], obs_dim)).to(args.device)

        for i in tqdm(range(args.num_episodes), desc="Inference Episodes"):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            with tqdm(total=1001, desc=f"Episode {i+1}/{args.num_episodes}", leave=False) as pbar:
                while not np.all(cum_done) and t < 1000 + 1:
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                    priors[0][:, :, 0] = obs.unsqueeze(1)

                    for j in range(n_levels):
                        if j == 0:
                            traj, log = diffusions[j].sample(
                                priors[j].view(-1, planning_horizons[j], obs_dim),
                                solver=args.solver,
                                n_samples=args.num_envs * args.num_candidates, 
                                sample_steps=args.sampling_steps,
                                use_ema=args.use_ema,
                                w_cg=args.task.w_cg,
                                temperature=args.temperature,
                                faiss_wrapper=faiss_wrapper, 
                                proj_range=args.task.proj_range,
                                n_manifold_samples=args.task.n_manifold_samples, tau=args.tau)

                            logp = log["log_p"]
                            traj = einops.rearrange(traj, "(b k) h d -> b k h d", k=args.num_candidates)
                            logp = einops.rearrange(logp, "(b k) 1 -> b k 1", k=args.num_candidates).squeeze(-1)
                            idx = torch.argmax(logp, dim=-1)
                            traj = traj[torch.arange(args.num_envs), idx]
                        else:
                            traj, log = diffusions[j].sample(
                                priors[j].view(-1, planning_horizons[j], obs_dim),
                                solver=args.solver,
                                n_samples=args.num_envs, 
                                sample_steps=args.sampling_steps,
                                use_ema=args.use_ema,
                                w_cg=args.task.w_cg,
                                temperature=args.temperature)

                        if j < n_levels - 1:
                            priors[j + 1][:, [0, -1]] = traj[:, [0, 1]]

                    with torch.no_grad():
                        act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

                    obs, rew, done, info = env_eval.step(act)

                    pbar.update(1)

                    t += 1
                    cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                    ep_reward += rew
                    print(f'[t={t}] xy: {obs[:, :2]}')
                    print(f'[t={t}] rew: {ep_reward}')

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