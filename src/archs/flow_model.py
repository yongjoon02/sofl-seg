"""Flow matching models for vessel segmentation."""
import autorootcwd  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.segmentation.dice import DiceScore
from torchdiffeq import odeint

from src.registry.base import ARCHS_REGISTRY
from src.archs.components import unet  # noqa: F401 - Register architectures
from src.archs.components.flow import SchrodingerBridgeConditionalFlowMatcher
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error
from src.archs.components.utils import random_patch_batch, select_patch_params


class _LossSummaryModule(nn.Module):
    """Lightweight module to expose loss configuration in Lightning logs."""

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description

    def forward(self, *args, **kwargs):  # pragma: no cover - not meant to be called
        raise RuntimeError("Loss summary module is not callable.")

    def extra_repr(self) -> str:
        return self.description

class FlowModel(L.LightningModule):
    """Lightning module for flow matching producing binary masks."""
    
    def __init__(
        self,
        arch_name: str = 'dhariwal_concat_unet',
        image_size: int = 512,
        patch_plan: list = [(320, 6), (384, 4), (416, 3), (512, 1)],
        dim: int = 32,
        timesteps: int = 15,
        sigma: float = 0.25,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        num_ensemble: int = 1,
        data_name: str = 'xca',
        log_image_enabled: bool = False,
        log_image_names: list = None,
        # UNet architecture parameters
        model_channels: int = 32,
        channel_mult: list = [1, 2, 4, 8],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: list = [16, 16, 8, 8],
        dropout: float = 0.0,
        label_dim: int = 0,
        augment_dim: int = 0,
        # Loss configuration
        loss_type: str = 'l2',  # 'l2', 'l2_bce', 'l2_l2', 'l2_bce_l2', 'l2_bce_dice' (also supports legacy 'l1*')
        bce_weight: float = 0.5,
        l2_weight: float = 0.1,
        dice_weight: float = 0.2,
        loss: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        
        # Get model class from registry
        if arch_name not in ARCHS_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHS_REGISTRY.keys())}")
        
        arch_class = ARCHS_REGISTRY.get(arch_name)
        
        self.use_geometry_head = arch_name == 'dhariwal_concat_unet_multihead'

        # For dhariwal_concat_unet, need mask_channels and input_img_channels
        if arch_name in {'dhariwal_concat_unet', 'dhariwal_concat_unet_multihead'}:
            self.unet = arch_class(
                img_resolution=image_size,
                mask_channels=1,  # geometry output
                input_img_channels=1,  # image condition
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dim=label_dim,
                augment_dim=augment_dim,
            )
        else:
            self.unet = arch_class(
                img_resolution=image_size,
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dim=label_dim,
                augment_dim=augment_dim,
            )

        self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        
        # Metrics
        self.val_metrics = MetricCollection({
            'dice': DiceScore(num_classes=num_classes, average='macro', include_background=False, input_format="index"),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'specificity': BinarySpecificity(),
            'iou': BinaryJaccardIndex(),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
        
        self.log_image_enabled = log_image_enabled
        self.log_image_names = log_image_names if log_image_names is not None else ['00036.png']
        
        # Loss configuration (registry 우선)
        from src.registry import LOSS_REGISTRY
        self.use_registry_loss = loss is not None
        if self.use_registry_loss:
            loss_name = loss.get('name')
            loss_params = loss.get('params', {})
            if loss_name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss: {loss_name}. Available: {LOSS_REGISTRY.keys()}")
            self.loss_fn = LOSS_REGISTRY.get(loss_name)(**loss_params)
            self.loss_description = f"{loss_name}({loss_params})"
        else:
            self.loss_type = loss_type
            self.bce_weight = bce_weight
            self.l2_weight = l2_weight
            self.dice_weight = dice_weight
            if 'dice' in loss_type:
                from src.losses import DiceLoss
                self.dice_loss = DiceLoss()
            self.loss_description = f"builtin:{loss_type}"

        # Register lightweight summary module so logs show loss info.
        self.loss_summary = _LossSummaryModule(self.loss_description)

        # Buffer for distributed image logging (validation/test).
        self._pending_image_logs: list[dict] = []

        # Sliding window inferer for validation/test (match diffusion behavior).
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )

    def _infer_sliding(self, images: torch.Tensor) -> torch.Tensor:
        """Run sliding-window inference with fresh noise per patch."""
        def _sample_fn(x: torch.Tensor) -> torch.Tensor:
            noise = torch.randn_like(x)
            return self.sample(noise, x)
        return self.inferer(images, _sample_fn)

    def training_step(self, batch, batch_idx):
        images = batch['image']  # condition
        # geometry: soft label/distance map 지원 (XCA에서는 geometry 없으면 label 사용)
        geometry = batch.get('geometry', batch.get('label'))  # target
        labels = batch.get('label', geometry)
        
        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)
        
        # Prepare noise (x0) and target geometry
        if not getattr(self.hparams, 'use_x0_mixing', False) or getattr(self.hparams, 'x0_policy', 'legacy') == 'legacy':
            noise = torch.randn_like(geometry)
        else:
            noise = self.make_x0_mixed(geometry, 'train', self.global_step)
        
        # Random patch extraction
        noise, geometry, images, labels = random_patch_batch(
            [noise, geometry, images, labels], patch_size, num_patches
        )
        
        # Flow matching: x (noise) -> geometry
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, geometry)

        with torch.no_grad():
            # Diagnostics: noise vs target scale and early-t visualization.
            x1 = geometry
            x0 = noise
            x1_norm = torch.linalg.vector_norm(x1.flatten(1), dim=1)
            x0_diff_norm = torch.linalg.vector_norm((x1 - x0).flatten(1), dim=1)
            ratio = (x0_diff_norm / (x1_norm + 1e-8)).mean()
            self.log('train/x1_x0_ratio', ratio, prog_bar=False, sync_dist=True)

            diag_every = getattr(self.hparams, 'log_diag_every_n_steps', 200)
            target_t = getattr(self.hparams, 'log_diag_t', 0.1)
            if (
                self.global_step % diag_every == 0
                and hasattr(self, 'logger')
                and self.logger is not None
                and hasattr(self.logger, 'experiment')
                and getattr(self.trainer, 'is_global_zero', True)
            ):
                idx = torch.argmin((t - target_t).abs()).item()
                xt_img = xt[idx, 0:1].detach()
                xt_min = xt_img.min()
                xt_max = xt_img.max()
                if (xt_max - xt_min) > 1e-6:
                    xt_img = (xt_img - xt_min) / (xt_max - xt_min)
                    self.logger.experiment.add_image(
                        f'train/xt_t{target_t:.2f}',
                        xt_img,
                        global_step=self.global_step,
                    )
        
        # UNet forward: (x=xt, time=t, cond=images)
        unet_out = self.unet(xt, t, images)
        if self.use_geometry_head:
            v = unet_out[:, 0:1, :, :]
            geometry_pred = unet_out[:, 1:2, :, :]
        else:
            v = unet_out
            geometry_pred = None
        
        # Compute loss based on loss_type
        if self.use_registry_loss:
            loss, loss_dict = self.loss_fn(
                v, ut, xt, geometry, t=t, geometry_pred=geometry_pred, hard_labels=labels
            )
            for name, value in loss_dict.items():
                self.log(f'train/{name}_loss', value, prog_bar=False, sync_dist=True)
        else:
            loss = self._compute_loss(v, ut, xt, geometry, t)
        
        # Log (sync_dist for DDP)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def _compute_loss(self, v, ut, xt, geometry, t):
        """Compute loss based on loss_type configuration."""
        losses = {}

        # Base flow matching loss:
        # Default is L2(MSE) to match standard flow-matching objective.
        # For backward compatibility, legacy 'l1*' loss_type keeps L1 base.
        use_l1_base = isinstance(self.loss_type, str) and self.loss_type.startswith('l1')
        if use_l1_base:
            base_loss = torch.abs(v - ut).mean()
            losses['l1'] = base_loss
        else:
            base_loss = ((v - ut) ** 2).mean()
            losses['l2'] = base_loss
        
        # Additional losses based on loss_type
        if self.loss_type in {'l1', 'l2'}:
            # Default: base loss only
            total_loss = base_loss
        
        elif self.loss_type in {'l1_bce', 'l2_bce'}:
            # Base + BCE (recommended for SAUNA soft labels)
            # Compute output geometry from xt for BCE loss
            # Approximate output: xt at t=1 should be close to geometry
            # Use xt as proxy for output geometry (or compute from flow)
            output_geometry = xt  # Simplified: use current state as proxy
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry
            
            # Convert to [0, 1] range if needed (SAUNA is already in [0, 1])
            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            
            # BCE loss on probabilities
            eps = 1e-7
            output_probs = torch.clamp(output_probs, eps, 1 - eps)
            bce_loss = -(target_probs * torch.log(output_probs) + 
                        (1 - target_probs) * torch.log(1 - output_probs)).mean()
            losses['bce'] = bce_loss
            
            total_loss = base_loss + self.bce_weight * bce_loss
        
        elif self.loss_type in {'l1_l2', 'l2_l2'}:
            # Base + additional L2 regularizer on (v - ut)
            # (Kept for backward compatibility/tuning.)
            l2_loss = ((v - ut) ** 2).mean()
            losses['l2'] = l2_loss
            total_loss = base_loss + self.l2_weight * l2_loss
        
        elif self.loss_type in {'l1_bce_l2', 'l2_bce_l2'}:
            # Base + BCE + additional L2 regularizer
            output_geometry = xt
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry
            
            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            
            eps = 1e-7
            output_probs = torch.clamp(output_probs, eps, 1 - eps)
            bce_loss = -(target_probs * torch.log(output_probs) + 
                        (1 - target_probs) * torch.log(1 - output_probs)).mean()
            l2_loss = ((v - ut) ** 2).mean()
            
            losses['bce'] = bce_loss
            losses['l2'] = l2_loss
            total_loss = base_loss + self.bce_weight * bce_loss + self.l2_weight * l2_loss
        
        elif self.loss_type in {'l1_bce_dice', 'l2_bce_dice'}:
            # Base + BCE + Dice
            output_geometry = xt
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry
            
            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            
            # BCE loss
            eps = 1e-7
            output_probs_clamped = torch.clamp(output_probs, eps, 1 - eps)
            bce_loss = -(target_probs * torch.log(output_probs_clamped) + 
                        (1 - target_probs) * torch.log(1 - output_probs_clamped)).mean()
            
            # Dice loss
            dice_loss = self.dice_loss(output_probs.unsqueeze(1), target_probs)
            
            losses['bce'] = bce_loss
            losses['dice'] = dice_loss
            total_loss = base_loss + self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        else:
            # Unknown loss_type, fallback to base loss
            total_loss = base_loss
        
        # Log individual losses
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}_loss', loss_value, prog_bar=False, sync_dist=True)
        
        return total_loss

    def make_x0_mixed(self, x1: torch.Tensor, stage: str, global_step: int) -> torch.Tensor:
        """Generate x0 by mixing near-start (blurred x1) and noise-start."""
        p_start = float(getattr(self.hparams, 'x0_p_start', 0.8))
        p_end = float(getattr(self.hparams, 'x0_p_end', 0.0))
        decay_steps = float(getattr(self.hparams, 'x0_p_decay_steps', 50000))
        x0_alpha = float(getattr(self.hparams, 'x0_alpha', 0.1))
        x0_sigma = float(getattr(self.hparams, 'x0_sigma', 0.1))
        blur_sigma = float(getattr(self.hparams, 'x0_blur_sigma', 4.0))

        if decay_steps <= 0:
            p = p_end
        else:
            progress = min(max(global_step / decay_steps, 0.0), 1.0)
            p = p_end + (p_start - p_end) * (1.0 - progress)

        if stage in {'val', 'test'} and getattr(self.hparams, 'debug_val_use_near_start', False):
            p = 1.0

        # Near-start: blur x1 then add small noise.
        if blur_sigma > 0:
            radius = int(torch.ceil(torch.tensor(3.0 * blur_sigma)).item())
            kernel_size = radius * 2 + 1
            device = x1.device
            dtype = x1.dtype
            coords = torch.arange(kernel_size, device=device, dtype=dtype) - radius
            kernel_1d = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
            kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
            channels = x1.shape[1]
            kernel_x = kernel_x.repeat(channels, 1, 1, 1)
            kernel_y = kernel_y.repeat(channels, 1, 1, 1)
            padding = (radius, radius, radius, radius)
            blurred = F.pad(x1, padding, mode='reflect')
            blurred = F.conv2d(blurred, kernel_x, groups=channels)
            blurred = F.conv2d(blurred, kernel_y, groups=channels)
        else:
            blurred = x1
        near = blurred + x0_alpha * torch.randn_like(x1)

        # Noise-start: pure Gaussian noise.
        noise = x0_sigma * torch.randn_like(x1)

        # Sample mixing mask per sample.
        if p <= 0:
            x0 = noise
            near_rate = 0.0
        elif p >= 1:
            x0 = near
            near_rate = 1.0
        else:
            mask = (torch.rand(x1.shape[0], device=x1.device) < p).float().view(-1, 1, 1, 1)
            x0 = mask * near + (1 - mask) * noise
            near_rate = mask.mean().item()

        if getattr(self.hparams, 'use_x0_mixing', False):
            self.log(f'{stage}/x0_near_rate', near_rate, prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x0_mean', x0.mean(), prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x0_std', x0.std(), prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x1_mean', x1.mean(), prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x1_std', x1.std(), prog_bar=False, sync_dist=True)

        return x0


    def sample(self, noise, images, return_intermediate: bool = False, save_steps: list = None):
        """Sample from flow matching model (inference).
        
        Args:
            noise: (B, 1, H, W) - initial noise
            images: (B, 1, H, W) - condition images
            return_intermediate: If True, return intermediate trajectory
            save_steps: List of step indices to save when return_intermediate=True
        
        Returns:
            If return_intermediate=False: output_geometry (B, 1, H, W)
            If return_intermediate=True, save_steps=None: (traj, output_geometry)
            If return_intermediate=True, save_steps=[...]: (saved_steps dict, output_geometry)
        """
        if save_steps is None:
            save_steps = [2, 4, 6, 8, 10, 12, 14]
        
        # Store images for ode_func
        self._sample_images = images
        
        traj = odeint(
            self.ode_func,
            noise,
            torch.linspace(0, 1, self.hparams.timesteps, device=noise.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5"
        )
        
        output_geometry = traj[-1]
        
        if return_intermediate:
            if save_steps is not None:
                saved_steps = {t: traj[t] for t in save_steps}
                return saved_steps, output_geometry
            return traj, output_geometry
        
        return output_geometry

    def validation_step(self, batch, batch_idx):
        images = batch['image']  # condition
        labels = batch['label']
        geometry = batch.get('geometry', batch.get('label'))  # target
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                output_geometry = self._infer_sliding(images)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            output_geometry = self._infer_sliding(images)
        
        # Compute reconstruction loss (final generation quality)
        loss = torch.abs(output_geometry - geometry).mean()
        self.log('val/reconstruction_loss', loss, prog_bar=True, sync_dist=True)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        
        # Apply sigmoid to convert x1_pred to probability
        output_prob = torch.sigmoid(output_geometry)
        preds = (output_prob > 0.5).long()  # threshold after sigmoid
        
        # Convert geometry for logging (ensure same dimensions as output_geometry)
        if geometry.dim() == 4 and geometry.shape[1] == 1:
            geometry = geometry.squeeze(1)
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log (sync_dist for DDP)
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True, sync_dist=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False, sync_dist=True)

        # Sanity check 단계에서는 이미지 로깅을 스킵해 불필요한 all_gather_object 오버헤드 제거
        if getattr(self.trainer, "sanity_checking", False):
            return general_metrics['dice']
        
        # Queue images for logging (handles DDP sharding safely).
        self._queue_images_for_logging(
            sample_names=sample_names,
            images=images,
            labels=labels,
            preds=preds,
            tag_prefix='val',
            geometry=geometry,
            output_geometry=output_geometry,
        )
        
        return general_metrics['dice']

    def on_validation_epoch_end(self):
        self._flush_queued_images()

    def test_step(self, batch, batch_idx):
        images = batch['image']  # condition
        labels = batch['label']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                output_geometry = self._infer_sliding(images)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            output_geometry = self._infer_sliding(images)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        
        # Apply sigmoid to convert x1_pred to probability
        output_prob = torch.sigmoid(output_geometry)
        preds = (output_prob > 0.5).long()  # threshold after sigmoid
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log (sync_dist for DDP)
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()}, sync_dist=True)
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()}, sync_dist=True)
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='test')
        
        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks_binary = (preds > 0).float()
            label_masks = (labels > 0).float()
            
            # Prepare metrics for each sample
            sample_metrics = []
            for i in range(images.shape[0]):
                sample_metric = {}
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in general_metrics.items()})
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in vessel_metrics.items()})
                sample_metrics.append(sample_metric)
            
            # Save predictions
            self.trainer.logger.save_predictions(
                sample_names, images, pred_masks_binary, label_masks, sample_metrics
            )
        
        return general_metrics['dice']

    @torch.no_grad()
    def ode_func(self, t, x):
        """ODE function for flow matching.
        
        Args:
            t: Time step (scalar or tensor)
            x: State tensor (B, 1, H, W) - noise/geometry at time t
        
        Returns:
            Velocity field from UNet
        """
        if isinstance(t, torch.Tensor):
            t = t.expand(x.shape[0])
        else:
            t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        
        # Get condition images (stored during sample)
        images = self._sample_images
        
        # UNet forward: (x=noise, time=t, cond=images)
        unet_out = self.unet(x, t, images)
        if self.use_geometry_head:
            return unet_out[:, 0:1, :, :]
        return unet_out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # val/dice는 높을수록 좋음
            patience=3,  # validation 주기(25 epoch)를 고려 = 75 epoch
            factor=0.5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/dice',  # ✅ 올바른 metric
                'interval': 'epoch',
                'frequency': 25,  # validation 주기와 일치
            }
        }



    def _log_images(
        self,
        sample_names,
        images,
        labels,
        preds,
        tag_prefix: str,
        geometry=None,
        output_geometry=None,
        **_,
    ):
        """Log a batch of images to TensorBoard (single-process helper)."""
        # Check if logging is enabled
        if not hasattr(self.hparams, 'log_image_enabled') or not self.hparams.log_image_enabled:
            return
        
        # DDP: only log on rank 0
        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return
        
        log_names = getattr(self.hparams, 'log_image_names', None)
        
        for i, name in enumerate(sample_names):
            filename = name.split('/')[-1] if '/' in name else name
            if log_names is not None and filename not in log_names:
                continue

            self._log_one_image(
                filename=filename,
                image=images[i],
                label=labels[i],
                pred=preds[i],
                tag_prefix=tag_prefix,
                geometry=(geometry[i] if geometry is not None else None),
                output_geometry=(output_geometry[i] if output_geometry is not None else None),
            )

    def _log_one_image(
        self,
        *,
        filename: str,
        image: torch.Tensor,
        label: torch.Tensor,
        pred: torch.Tensor,
        tag_prefix: str,
        geometry: torch.Tensor | None = None,
        output_geometry: torch.Tensor | None = None,
    ) -> None:
        """Log a single sample with separate panels (readable in TensorBoard)."""
        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return

        def ensure_chw(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                return x.unsqueeze(0)
            return x

        img = ensure_chw((image + 1) / 2).clamp(0.0, 1.0)
        lab = ensure_chw(label.float()).clamp(0.0, 1.0)
        prd = ensure_chw(pred.float()).clamp(0.0, 1.0)

        base = f"{tag_prefix}/{filename}"
        self.logger.experiment.add_image(f"{base}/image", img, self.global_step)
        self.logger.experiment.add_image(f"{base}/label", lab, self.global_step)
        if geometry is not None:
            geo = ensure_chw(geometry.float()).clamp(0.0, 1.0)
            self.logger.experiment.add_image(f"{base}/geometry", geo, self.global_step)
        if output_geometry is not None:
            out_geo = ensure_chw(output_geometry.float()).clamp(0.0, 1.0)
            self.logger.experiment.add_image(f"{base}/output_geometry", out_geo, self.global_step)
        self.logger.experiment.add_image(f"{base}/pred", prd, self.global_step)

    def _queue_images_for_logging(
        self,
        *,
        sample_names,
        images: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        tag_prefix: str,
        geometry: torch.Tensor | None = None,
        output_geometry: torch.Tensor | None = None,
    ) -> None:
        """Queue selected images for logging; safe under DDP sharding."""
        if not hasattr(self.hparams, 'log_image_enabled') or not self.hparams.log_image_enabled:
            return

        log_names = getattr(self.hparams, 'log_image_names', None)
        if log_names is None:
            return

        for i, name in enumerate(sample_names):
            filename = name.split('/')[-1] if '/' in name else name
            if filename not in log_names:
                continue

            self._pending_image_logs.append(
                {
                    'tag_prefix': tag_prefix,
                    'filename': filename,
                    'image': images[i].detach().cpu(),
                    'label': labels[i].detach().cpu(),
                    'pred': preds[i].detach().cpu(),
                    'geometry': (geometry[i].detach().cpu() if geometry is not None else None),
                    'output_geometry': (output_geometry[i].detach().cpu() if output_geometry is not None else None),
                }
            )

    def _flush_queued_images(self) -> None:
        """Gather queued images across ranks and log them once on rank 0."""
        if not self._pending_image_logs:
            return

        gathered: list[list[dict]] | None = None
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, self._pending_image_logs)
        else:
            gathered = [self._pending_image_logs]

        # Clear local buffer ASAP to avoid growth if something goes wrong later.
        self._pending_image_logs = []

        # Only log on rank 0 (logger exists only there in this runner).
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return

        # Flatten and de-duplicate by filename (DDP may still duplicate in some setups).
        flat: list[dict] = []
        for part in gathered:
            if part:
                flat.extend(part)

        seen = set()
        for item in flat:
            key = (item.get('tag_prefix'), item.get('filename'))
            if key in seen:
                continue
            seen.add(key)
            self._log_one_image(
                filename=item['filename'],
                image=item['image'],
                label=item['label'],
                pred=item['pred'],
                tag_prefix=item['tag_prefix'],
                geometry=item.get('geometry'),
                output_geometry=item.get('output_geometry'),
            )
