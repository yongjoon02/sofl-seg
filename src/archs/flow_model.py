"""Flow matching models for vessel segmentation."""
import autorootcwd  # noqa: F401
import torch
import torch.nn as nn
import torch.distributed as dist
import lightning.pytorch as L
from torchmetrics import MetricCollection
from torchdiffeq import odeint

from src.registry.base import ARCHS_REGISTRY
from src.archs.components import unet  # noqa: F401 - Register architectures
from src.archs.components.flow import SchrodingerBridgeConditionalFlowMatcher
from src.metrics.general_metrics import Dice, Precision, Recall, Specificity, JaccardIndex
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

class FlowCoordModel(L.LightningModule):
    """Lightning module for coordinate-aware flow matching producing binary masks."""
    
    def __init__(
        self,
        arch_name: str = 'dhariwal_unet_4channel',
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
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })
        
        self.log_image_enabled = log_image_enabled
        self.log_image_names = log_image_names if log_image_names is not None else ['00036.png']

    def training_step(self, batch, batch_idx):
        images = batch['image']  # condition
        labels = batch['label']
        # geometry: soft label/distance map 지원 (향후 확장용)
        geometry = batch['geometry']  # target
        
        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)
        
        # Prepare noise (x0) and target geometry (x1)
        noise = torch.randn_like(geometry)
        
        # Random patch extraction
        noise, geometry, images = random_patch_batch(
            [noise, geometry, images], patch_size, num_patches
        )
        
        # Flow matching: noise (x0) -> geometry (x1)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, geometry)
        
        # UNet forward: (x=xt, time=t, cond=images)
        # dhariwal_concat_unet expects: forward(x, time, cond)
        v = self.unet(xt, t, images)  # xt: noisy geometry, images: condition
        
        # Compute loss
        loss = torch.abs(v - ut).mean()
        
        # Log
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def sample(self, input_4ch, return_intermediate: bool = False, save_steps: list = None):
        """Sample from flow matching model (inference).
        
        Args:
            input_4ch: (B, 4, H, W) = [image, noise, coordx, coordy]
            return_intermediate: If True, return intermediate trajectory
            save_steps: List of step indices to save when return_intermediate=True
        
        Returns:
            If return_intermediate=False: output_geometry (B, 1, H, W)
            If return_intermediate=True, save_steps=None: (traj, output_geometry)
            If return_intermediate=True, save_steps=[...]: (saved_steps dict, output_geometry)
        """
        if save_steps is None:
            save_steps = [2, 4, 6, 8, 10, 12, 14]
        
        traj = odeint(
            self.ode_func,
            input_4ch,
            torch.linspace(0, 1, self.hparams.timesteps, device=input_4ch.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5"
        )
        
        output_4ch = traj[-1]
        output_geometry = output_4ch[:, 1:2, :, :]
        
        if return_intermediate:
            if save_steps is not None:
                saved_steps = {t: traj[t][:, 1:2] for t in save_steps}
                return saved_steps, output_geometry
            return traj, output_geometry
        
        return output_geometry

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        geometry = batch['geometry']
        coordinate = batch['coordinate']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Prepare input
        coordx = coordinate[:, 0:1]
        coordy = coordinate[:, 1:2]
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(images)
                input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
                saved_steps, output_geometry = self.sample(input_4ch, return_intermediate=True)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            noise = torch.randn_like(images)
            input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
            saved_steps, output_geometry = self.sample(input_4ch, return_intermediate=True)
        
        # Compute reconstruction loss (final generation quality)
        loss = torch.abs(output_geometry - geometry).mean()
        self.log('val/reconstruction_loss', loss, prog_bar=True)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.5).long()  # threshold=0.5 (geometry는 [0, 1] 범위)
        
        # Convert geometry for logging (ensure same dimensions as output_geometry)
        if geometry.dim() == 4 and geometry.shape[1] == 1:
            geometry = geometry.squeeze(1)
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)
        
        # Log images: include geometry if soft labels are used (geometry != labels)
        # Check if geometry is different from labels (soft label case)
        use_soft_label = not torch.equal(geometry, labels.float())
        if use_soft_label:
            self._log_images(sample_names, images, labels, preds, tag_prefix='val', 
                           geometry=geometry, output_geometry=output_geometry)
        else:
            self._log_images(sample_names, images, labels, preds, tag_prefix='val')
        
        return general_metrics['dice']

    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        coordinate = batch['coordinate']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        # Prepare input
        coordx = coordinate[:, 0:1]
        coordy = coordinate[:, 1:2]
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(images)
                input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
                output_geometry = self.sample(input_4ch)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            noise = torch.randn_like(images)
            input_4ch = torch.cat([images, noise, coordx, coordy], dim=1)
            output_geometry = self.sample(input_4ch)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.5).long()  # threshold=0.5 (geometry는 [0, 1] 범위)
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})
        
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
            x: State tensor (B, 4, H, W)
        
        Returns:
            Velocity field from UNet
        """
        if isinstance(t, torch.Tensor):
            t = t.expand(x.shape[0])
        else:
            t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        return self.unet(x, t, class_labels=None)

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



    def _log_images(self, sample_names, images, labels, preds, tag_prefix: str, geometry=None, output_geometry=None):
        """Log images to TensorBoard similar to supervised model.
        
        Args:
            sample_names: List of sample names
            images: Input images (B, C, H, W)
            labels: Binary labels (B, H, W)
            preds: Binary predictions (B, H, W)
            tag_prefix: Prefix for TensorBoard tag
            geometry: Optional soft label/geometry (B, H, W) - for soft label visualization
            output_geometry: Optional output geometry (B, H, W) - for soft label visualization
        """
        # Check if logging is enabled
        if not hasattr(self.hparams, 'log_image_enabled') or not self.hparams.log_image_enabled:
            return
        
        # DDP: only log on rank 0
        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return
        
        log_names = getattr(self.hparams, 'log_image_names', None)
        
        for i, name in enumerate(sample_names):
            filename = name.split('/')[-1] if '/' in name else name
            
            # Only log if filename matches log_image_names (or log all if not specified)
            if log_names is None or filename in log_names:
                print(f"[DEBUG] Logging image: {filename} at epoch {self.current_epoch}")
                img = (images[i] + 1) / 2
                pred = preds[i].float().unsqueeze(0)
                label = labels[i].float().unsqueeze(0)

                # Standard visualization: [image, label, pred]
                vis_row = torch.cat([img, label, pred], dim=-1)
                self.logger.experiment.add_image(
                    tag=f'{tag_prefix}/{filename}',
                    img_tensor=vis_row,
                    global_step=self.global_step,
                )

                # If geometry is provided (soft label), also log geometry comparison
                if geometry is not None and output_geometry is not None:
                    geom = geometry[i].float().unsqueeze(0)
                    out_geom = output_geometry[i].float().unsqueeze(0)

                    # Geometry visualization: [image, target_geometry, output_geometry]
                    geom_vis_row = torch.cat([img, geom, out_geom], dim=-1)
                    self.logger.experiment.add_image(
                        tag=f'{tag_prefix}_geometry/{filename}',
                        img_tensor=geom_vis_row,
                        global_step=self.global_step,
                    )


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
        
        # For dhariwal_concat_unet, need mask_channels and input_img_channels
        if arch_name == 'dhariwal_concat_unet':
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
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
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

    def training_step(self, batch, batch_idx):
        images = batch['image']  # condition
        # geometry: soft label/distance map 지원 (XCA에서는 geometry 없으면 label 사용)
        geometry = batch.get('geometry', batch.get('label'))  # target
        
        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)
        
        # Prepare noise (x0) and target geometry
        noise = torch.randn_like(geometry)
        
        # Random patch extraction
        noise, geometry, images = random_patch_batch(
            [noise, geometry, images], patch_size, num_patches
        )
        
        # Flow matching: x (noise) -> geometry
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(noise, geometry)
        
        # UNet forward: (x=xt, time=t, cond=images)
        v = self.unet(xt, t, images)
        
        # Compute loss based on loss_type
        if self.use_registry_loss:
            loss, loss_dict = self.loss_fn(v, ut, xt, geometry)
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
                noise = torch.randn_like(geometry)
                saved_steps, output_geometry = self.sample(noise, images, return_intermediate=True)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            noise = torch.randn_like(geometry)
            saved_steps, output_geometry = self.sample(noise, images, return_intermediate=True)
        
        # Compute reconstruction loss (final generation quality)
        loss = torch.abs(output_geometry - geometry).mean()
        self.log('val/reconstruction_loss', loss, prog_bar=True, sync_dist=True)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.5).long()  # threshold=0.5 (geometry는 [0, 1] 범위)
        
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
        
        # Prepare noise (same shape as geometry would be)
        noise = torch.randn(images.shape[0], 1, images.shape[2], images.shape[3], 
                          device=images.device, dtype=images.dtype)
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                noise = torch.randn_like(noise)
                output_geometry = self.sample(noise, images)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            output_geometry = self.sample(noise, images)
        
        # Convert predictions to class indices
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        preds = (output_geometry > 0.5).long()  # threshold=0.5 (geometry는 [0, 1] 범위)
        
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
        return self.unet(x, t, images)

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
