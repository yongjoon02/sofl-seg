"""Cold diffusion segmentation models.

Cold Diffusion replaces Gaussian noise with deterministic degradation
from segmentation -> image, then restores image -> segmentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.archs.components.diffusion_unet import (
    extract,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    MedSegDiffUNet,
)
from src.archs.components.gaussian_diffusion import GaussianDiffusionModel, BceDiceLoss


def _mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ColdDiffusionModel(GaussianDiffusionModel):
    """Cold Diffusion: deterministic degradation/restoration for segmentation."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        objective: str = "predict_x0",
        beta_schedule: str = "cosine",
        loss_type: str = "mse",
    ) -> None:
        super().__init__(model, timesteps, sampling_timesteps, objective, beta_schedule, loss_type)
        self.bce_dice_loss = BceDiceLoss(wb=1.0, wd=1.0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward degradation: blend segmentation towards conditional image."""
        alpha_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        beta_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return alpha_t * x_start + beta_t * noise

    def forward(self, seg_mask: torch.Tensor, cond_img: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """Cold diffusion loss (mse or hybrid)."""
        device = self.device
        seg_mask, cond_img = seg_mask.to(device), cond_img.to(device)

        b = seg_mask.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        seg_mask = normalize_to_neg_one_to_one(seg_mask)
        x_degraded = self.q_sample(x_start=seg_mask, t=times, noise=cond_img)
        predicted_seg = self.model(x_degraded, times, cond_img)

        if self.loss_type == "mse":
            return F.mse_loss(predicted_seg, seg_mask)

        if self.loss_type == "hybrid":
            time_weights = torch.pow((times.float() + 1) / self.num_timesteps, gamma)
            mse_loss = _mean_flat((predicted_seg - seg_mask) ** 2)
            weighted_mse = (time_weights * mse_loss).mean()

            pred_seg = unnormalize_to_zero_to_one(torch.clamp(predicted_seg, min=-1.0, max=1.0))
            target_seg = unnormalize_to_zero_to_one(seg_mask)
            bce_dice = self.bce_dice_loss(pred_seg, target_seg)

            alpha, beta = 0.1, 0.9
            return alpha * weighted_mse + beta * bce_dice

        raise ValueError(f"Unknown loss type: {self.loss_type}")

    @torch.no_grad()
    def sample(self, cond_img: torch.Tensor, save_steps: list[int] | None = None):
        """Reverse process: image -> segmentation (deterministic)."""
        cond_img = cond_img.to(self.device)
        b, _, h, w = cond_img.shape

        img = cond_img
        saved_steps = {}
        save_set = set(save_steps) if save_steps is not None else None

        for t in reversed(range(0, self.num_timesteps)):
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            preds = self.model_predictions(img, batched_times, cond_img, clip_x_start=True)
            pred_x0 = preds.predict_x_start

            if save_set is not None and t in save_set:
                saved_steps[t] = unnormalize_to_zero_to_one(pred_x0).cpu()

            if t > 0:
                batched_times_prev = torch.full((b,), t - 1, device=self.device, dtype=torch.long)
                img = self.q_sample(x_start=pred_x0, t=batched_times_prev, noise=cond_img)
            else:
                img = pred_x0

        img = unnormalize_to_zero_to_one(img)
        if save_set is not None:
            return {"final": img, "steps": saved_steps}
        return img


def create_colddiff(
    image_size: int = 224,
    dim: int = 24,
    timesteps: int = 50,
    loss_type: str = "hybrid",
    conditioning_fmap_size: int | None = None,
    channel_mult: tuple[int, ...] = (1, 2, 3, 4),
) -> ColdDiffusionModel:
    """Factory for cold diffusion segmentation."""
    unet = MedSegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=channel_mult,
        full_self_attn=(False, False, True, True),
        mid_transformer_depth=1,
        conditioning_fmap_size=conditioning_fmap_size,
    )
    return ColdDiffusionModel(
        unet,
        timesteps=timesteps,
        objective="predict_x0",
        beta_schedule="cosine",
        loss_type=loss_type,
    )
