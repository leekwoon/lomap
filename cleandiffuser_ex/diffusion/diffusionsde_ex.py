from typing import Optional, Union, Callable

import numpy as np
import torch
from cleandiffuser.utils import SUPPORTED_SAMPLING_STEP_SCHEDULE
from cleandiffuser.diffusion.diffusionsde import (
    at_least_ndim, epstheta_to_xtheta, xtheta_to_epstheta,
    DiscreteDiffusionSDE, ContinuousDiffusionSDE, SUPPORTED_SOLVERS)

from cleandiffuser_ex.local_manifold import LocalManifold


class DiscreteDiffusionSDEEX(DiscreteDiffusionSDE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_manifold = LocalManifold(device=self.device)

    @torch.no_grad()
    def get_manifold_samples(
            self, xt, t, 
            n_manifold_samples, prior, faiss_wrapper):
        
        batch_size, *dims = xt.shape

        if t[0].item() == 0:
            x0_t = xt
            x0_t_flat = x0_t.reshape(batch_size, -1)

            # 3) For each batch item => KNN in FAISS
            #    We'll do a single batched call => shape (batch_size, k)
            distances, idxs = faiss_wrapper.search(
                x0_t_flat.cpu().numpy(), n_manifold_samples)
            
            x0_t_flat_neighbors = faiss_wrapper.get_original_vectors(idxs)  # numpy array
            x0_t_flat_neighbors = torch.from_numpy(x0_t_flat_neighbors).to(self.device)
            x0_t_neighbors = x0_t_flat_neighbors.reshape(batch_size, n_manifold_samples, *dims)
            return x0_t_neighbors
        else:
            alpha_t = at_least_ndim(self.alpha[t], xt.dim())
            sigma_t = at_least_ndim(self.sigma[t], xt.dim())

            # Use Tweedie formula to get x_{0|t} from x_t.
            pred = self.model_ema["diffusion"](xt, t, None)
            x0_t = pred if not self.predict_noise else epstheta_to_xtheta(xt, alpha_t, sigma_t, pred)
            # fix the known portion
            x0_t = x0_t * (1. - self.fix_mask) + prior * self.fix_mask

            # 2) Flatten x_{0|t}
            x0_t_flat = x0_t.reshape(batch_size, -1)

            # 3) For each batch item => KNN in FAISS
            #    We'll do a single batched call => shape (batch_size, k)
            distances, idxs = faiss_wrapper.search(
                x0_t_flat.cpu().numpy(), n_manifold_samples)

            # get the original x0 neighbors => shape (batch_size, n_manifold_samples, dim)
            # original_data_flattened
            x0_t_flat_neighbors = faiss_wrapper.get_original_vectors(idxs)  # numpy array
            x0_t_flat_neighbors = torch.from_numpy(x0_t_flat_neighbors).to(self.device)
            x0_t_neighbors = x0_t_flat_neighbors.reshape(batch_size, n_manifold_samples, *dims)

            alpha_t = at_least_ndim(self.alpha[t], x0_t_neighbors.dim())
            sigma_t = at_least_ndim(self.sigma[t], x0_t_neighbors.dim())
            xt_neighbors = x0_t_neighbors * alpha_t + sigma_t * torch.randn_like(x0_t_neighbors)

            # Create the repeat pattern dynamically
            repeat_pattern = [1, n_manifold_samples] + [1] * len(dims)        
            prior = prior.unsqueeze(1).repeat(*repeat_pattern)  # Shape: (batch_size, n_manifold_samples, *dims)
            # fix the known portion
            xt_neighbors = xt_neighbors * (1. - self.fix_mask) + prior * self.fix_mask
            return xt_neighbors

    def compute_rg(
            self, xt, t, model, prior,
            forward_level: float = 0.8, n_mc_samples: int = 1):

        if not self.predict_noise:
            x0 = model["diffusion"](xt, t, None)
        else:
            raise NotImplementedError

        x0 = x0 * (1. - self.fix_mask) + prior * self.fix_mask

        rglb_samples = torch.zeros((xt.shape[0], n_mc_samples), device=self.device)
        for i in range(n_mc_samples):
            diffusion_steps = int(forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt_hat = x0 * fwd_alpha + fwd_sigma * torch.randn_like(x0)
            xt_hat = xt_hat * (1. - self.fix_mask) + prior * self.fix_mask

            t_hat = torch.full((xt_hat.shape[0],), diffusion_steps, dtype=torch.long, device=self.device)
            if not self.predict_noise:
                x0_hat = model["diffusion"](xt_hat, t_hat, None)
            else:
                raise NotImplementedError
            x0_hat = x0_hat * (1. - self.fix_mask) + prior * self.fix_mask

            diff = x0 - x0_hat.detach()
            rglb_sample = diff.reshape(diff.shape[0], -1).norm(p=2.0, dim=1)

            rglb_samples[:, i] = rglb_sample.view(-1)

        rglb = rglb_samples.mean(dim=-1)
        return rglb

    def low_density_guidance(
            self, xt, t, alpha, sigma, model, w,
            forward_level, n_mc_samples, prior, pred):

        if w == 0.0:
            return pred
        else:
            with torch.enable_grad():
                xt = xt.detach().requires_grad_(True)
                rg = self.compute_rg(xt, t, model, prior, 
                    forward_level=forward_level, n_mc_samples=n_mc_samples)
                grad = torch.autograd.grad(rg.sum(), xt)[0]

            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

            return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            # ----------- Low Density Guidance Params ------------ #
            w_ldg: float = 0.0,
            rg_forward_level: float = 0.8,
            n_mc_samples: int = 1,
            prior: torch.Tensor = None,
        ):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        pred = self.low_density_guidance(
            xt, t, alpha, sigma, model, w_ldg, rg_forward_level, 
            n_mc_samples, prior, pred)

        return pred, logp

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform",
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,
            # ----------- Warm-Starting -----------
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            # ---------Manifold Preserved Guidance -------- # 
            faiss_wrapper=None, # faiss_wrapper: A pre-built FAISS index containing the flattened x0 dataset
            proj_range=[],
            proj_mask: Optional[torch.Tensor] = None, 
            n_manifold_samples: int = 5,
            tau: float = 0.95, 
            # ----------- Low-Density Guidance -----------
            w_ldg: float = 0.0,
            rg_forward_level: float = 0.8,
            n_mc_samples: int = 1,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor):
            diffusion_steps = int(warm_start_forward_level * self.diffusion_steps)
            fwd_alpha, fwd_sigma = self.alpha[diffusion_steps], self.sigma[diffusion_steps]
            xt = warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            diffusion_steps = self.diffusion_steps
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    diffusion_steps, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(diffusion_steps, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas = self.alpha[sample_step_schedule]
        sigmas = self.sigma[sample_step_schedule]
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.long, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad,
                w_ldg, rg_forward_level, n_mc_samples, prior)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # noise & data prediction
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = ((alphas[i - 1] / alphas[i]) * xt -
                      2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta +
                      sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * D +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))
                else:
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            if faiss_wrapper and proj_range:
                if (sample_step_schedule[i] > 0) \
                    and (sample_step_schedule[i] >= int(proj_range[0] * (sample_steps - 1))) \
                    and (sample_step_schedule[i] <= int(proj_range[1] * (sample_steps - 1))):

                    assert proj_range[0] < proj_range[1]
                    # fix the known portion, and preserve the sampling history
                    xt = xt * (1. - self.fix_mask) + prior * self.fix_mask

                    xt_orig = xt.clone()

                    # Generate manifold samples
                    manifold_samples = self.get_manifold_samples(
                        xt,
                        t - 1,
                        n_manifold_samples, prior, faiss_wrapper)

                    # Compute PCA using LocalManifold 
                    self.local_manifold.compute_pca(manifold_samples, tau)
                    # Projection
                    xt_projected = self.local_manifold.project_points(xt)

                    if proj_mask is not None:
                        # Ensure proj_mask has correct device & shape
                        proj_mask = proj_mask.to(xt.device)
                        # proj_mask: True -> keep projected, False -> revert to orig
                        xt = torch.where(
                            proj_mask.view(-1, *([1]*(xt.dim()-1))),
                            xt_projected,
                            xt_orig
                        )
                    else:
                        xt = xt_projected

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log

