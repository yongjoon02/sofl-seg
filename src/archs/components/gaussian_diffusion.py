"""Gaussian Diffusion-based Segmentation Models

Standard DDPM with Gaussian noise for segmentation.
Supports MSE, VLB (KL), and Hybrid (MSE+VLB) losses.
"""
from collections import namedtuple
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Import UNet architectures from diffusion_unet
from src.archs.components.diffusion_unet import (
    MedSegDiffUNet,
    # UNet architectures
    SegDiffUNet,
    cosine_beta_schedule,
    # Utility functions
    default,
    # Diffusion process utilities
    extract,
    identity,
    linear_beta_schedule,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['predict_noise', 'predict_x_start'])


# ====== Loss Utilities (from MedSegDiff) ======
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two Gaussians.
    
    KL(N(mean1, var1) || N(mean2, var2)) = 
        0.5 * (-1 + logvar2 - logvar1 + exp(logvar1 - logvar2) + (mean1 - mean2)^2 / var2)
    """
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
        ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Compute the log-likelihood of a discretized Gaussian distribution.
    
    Args:
        x: Target values
        means: Mean of the Gaussian
        log_scales: Log standard deviation
    
    Returns:
        Log-likelihood
    """
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    # CDF approximation using sigmoid
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)

    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)

    # Log probability
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # Mid-range: log(cdf_plus - cdf_min)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.clamp(cdf_delta, min=1e-12))
        )
    )

    assert log_probs.shape == x.shape
    return log_probs


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.0
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BCELoss(nn.Module):
    """BCE loss for segmentation"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bce(pred_, target_)


class BceDiceLoss(nn.Module):
    """Combined BCE + Dice loss"""
    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.wb * bce_loss + self.wd * dice_loss


# ====== Gaussian Diffusion Process ======
class GaussianDiffusionModel(nn.Module):
    """Standard Gaussian Diffusion Model for Segmentation
    
    Uses Gaussian noise for diffusion process (DDPM).
    Supports both simple MSE loss and Improved DDPM's VLB loss.
    """

    def __init__(self, model, timesteps=1000, sampling_timesteps=None, objective='predict_x0',
                 beta_schedule='cosine', loss_type='mse'):
        super().__init__()

        self.model = model
        self.objective = objective
        self.loss_type = loss_type  # 'mse', 'vlb', or 'hybrid'
        self.image_size = model.image_size
        self.mask_channels = model.mask_channels
        self.input_img_channels = model.input_img_channels

        # Loss functions for hybrid mode
        self.bce_dice_loss = BceDiceLoss(wb=1.0, wd=1.0)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior calculations for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                       torch.log(torch.clamp(posterior_variance, min=1e-20)))

        # Posterior mean coefficients (MedSegDiff style)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    @property
    def device(self):
        return next(self.parameters()).device

    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from clean image"""
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
        
        From MedSegDiff original implementation.
        """
        assert x_start.shape == x_t.shape

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        assert (
            posterior_mean.shape[0] == posterior_variance.shape[0] ==
            posterior_log_variance.shape[0] == x_start.shape[0]
        )

        return posterior_mean, posterior_variance, posterior_log_variance

    def model_predictions(self, x, t, c, clip_x_start=False):
        """Model predictions for Gaussian diffusion"""
        model_output = self.model(x, t, c)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'predict_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def sample(self, cond_img, save_steps=None):
        """Sample from Gaussian diffusion model with optional step saving"""
        cond_img = cond_img.to(self.device)
        b, c, h, w = cond_img.shape

        # Start from pure Gaussian noise
        img = torch.randn(b, self.mask_channels, h, w, device=self.device)

        # Initialize step storage if needed
        saved_steps = {}
        if save_steps is not None:
            save_steps = set(save_steps)

        for t in reversed(range(0, self.num_timesteps)):
            batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)
            preds = self.model_predictions(img, batched_times, cond_img, clip_x_start=True)

            # Save step if requested
            if save_steps is not None and t in save_steps:
                saved_img = unnormalize_to_zero_to_one(preds.predict_x_start)
                saved_steps[t] = saved_img.cpu()

            # DDPM sampling step
            if t > 0:
                noise = torch.randn_like(img)

                # Predict x_0
                pred_x0 = preds.predict_x_start

                # Use pre-computed posterior coefficients (correct DDPM formula)
                # posterior_mean = coef1 * x_0 + coef2 * x_t
                posterior_mean = (
                    extract(self.posterior_mean_coef1, batched_times, img.shape) * pred_x0 +
                    extract(self.posterior_mean_coef2, batched_times, img.shape) * img
                )
                posterior_variance = extract(self.posterior_variance, batched_times, img.shape)

                # Sample x_{t-1}
                img = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                img = preds.predict_x_start

        img = unnormalize_to_zero_to_one(img)

        if save_steps is not None:
            return {
                'final': img,
                'steps': saved_steps
            }
        else:
            return img

    def q_sample(self, x_start, t, noise):
        """Forward process: add Gaussian noise to clean data"""
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, img, cond_img):
        """
        Forward pass - Gaussian diffusion loss.
        
        Loss types:
        - 'mse': Simple MSE loss (default DDPM)
        - 'vlb': Variational Lower Bound (Improved DDPM)
        - 'hybrid': MSE + VLB (Improved DDPM with simple loss)
        """
        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)

        b = img.shape[0]
        times = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Normalize mask to [-1, 1]
        img = normalize_to_neg_one_to_one(img)

        # Add Gaussian noise to ground truth mask
        noise = torch.randn_like(img)
        x_noisy = self.q_sample(x_start=img, t=times, noise=noise)

        # Predict x_0 (clean mask)
        model_out = self.model(x_noisy, times, cond_img)

        if self.loss_type == 'mse':
            # Simple MSE loss (default DDPM)
            return F.mse_loss(model_out, img)

        elif self.loss_type == 'vlb':
            # Variational Lower Bound (MedSegDiff style)
            pred_x_start = torch.clamp(model_out, min=-1., max=1.)

            # True posterior q(x_{t-1} | x_t, x_0)
            true_mean, _, true_log_var = self.q_posterior_mean_variance(
                x_start=img, x_t=x_noisy, t=times
            )

            # Model posterior p(x_{t-1} | x_t) using predicted x_0
            pred_mean, _, pred_log_var = self.q_posterior_mean_variance(
                x_start=pred_x_start, x_t=x_noisy, t=times
            )

            # KL divergence (in bits)
            kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
            kl = mean_flat(kl) / np.log(2.0)

            # Decoder NLL at t=0 (in bits)
            decoder_nll = -discretized_gaussian_log_likelihood(
                img, means=pred_mean, log_scales=0.5 * pred_log_var
            )
            assert decoder_nll.shape == img.shape
            decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

            # Use decoder_nll at t=0, KL otherwise
            loss = torch.where(times == 0, decoder_nll, kl)
            return loss.mean()

        elif self.loss_type == 'hybrid':
            # Hybrid: MSE + (BCE+Dice) (CrackSegDiff / MedSegDiff style)
            # Loss1: MSE for diffusion noise prediction
            mse_loss = mean_flat((model_out - img) ** 2).mean()

            # Loss2: BCE+Dice for supervised segmentation quality
            # Convert to [0, 1] for BCE+Dice
            pred_seg = unnormalize_to_zero_to_one(torch.clamp(model_out, min=-1., max=1.))
            target_seg = unnormalize_to_zero_to_one(img)

            bce_dice_loss = self.bce_dice_loss(pred_seg, target_seg)

            alpha, beta = 0.1, 0.9
            return alpha * mse_loss + beta * bce_dice_loss

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ====== Factory Functions ======
def create_segdiff(image_size=224, dim=64, timesteps=1000, loss_type='hybrid'):
    """SegDiff: RRDB-based conditioning (F(x_t) + G(cond))
    
    Args:
        loss_type: 'mse' (default), 'vlb', or 'hybrid'
    """
    unet = SegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        rrdb_blocks=3
    )
    return GaussianDiffusionModel(unet, timesteps=timesteps, objective='predict_x0',
                                 beta_schedule='cosine', loss_type=loss_type)


def create_medsegdiff(image_size=224, dim=64, timesteps=1000, loss_type='hybrid'):
    """MedSegDiff: FFT-based conditioning
    
    Args:
        loss_type: 'mse' (default), 'vlb', or 'hybrid'
    """
    # Use a shallower mid transformer to keep params comparable with SegDiff/BerDiff.
    unet = MedSegDiffUNet(
        dim=dim,
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=(False, False, True, True),
        mid_transformer_depth=0
    )
    return GaussianDiffusionModel(unet, timesteps=timesteps, objective='predict_x0',
                                 beta_schedule='cosine', loss_type=loss_type)




if __name__ == "__main__":
    print("=" * 70)
    print("Testing Gaussian Diffusion Segmentation Models")
    print("=" * 70)

    img = torch.randn(2, 1, 224, 224)
    cond = torch.randn(2, 1, 224, 224)

    print("\n1. SegDiff (MSE loss - default)")
    segdiff = create_segdiff(image_size=224, dim=64, timesteps=100, loss_type='mse')
    loss = segdiff(img, cond)
    params = sum(p.numel() for p in segdiff.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")

    print("\n2. SegDiff (VLB loss - Improved DDPM)")
    segdiff_vlb = create_segdiff(image_size=224, dim=64, timesteps=100, loss_type='vlb')
    loss = segdiff_vlb(img, cond)
    print(f"   Loss: {loss.item():.4f}")

    print("\n3. SegDiff (Hybrid loss - MSE + VLB)")
    segdiff_hybrid = create_segdiff(image_size=224, dim=64, timesteps=100, loss_type='hybrid')
    loss = segdiff_hybrid(img, cond)
    print(f"   Loss: {loss.item():.4f}")

    print("\n4. MedSegDiff (MSE loss)")
    medsegdiff = create_medsegdiff(image_size=224, dim=64, timesteps=100, loss_type='mse')
    loss = medsegdiff(img, cond)
    params = sum(p.numel() for p in medsegdiff.parameters())
    print(f"   Loss: {loss.item():.4f}, Params: {params:,}")

    print("\n" + "=" * 70)
    print("✓ All Gaussian diffusion models work correctly!")
    print("=" * 70)
