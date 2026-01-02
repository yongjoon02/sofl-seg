"""Diffusion models for vessel segmentation.
Based on supervised_model.py structure with MedSegDiff and BerDiff.
"""
from copy import deepcopy

import lightning.pytorch as L
import torch
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection

from src.metrics import (
    Betti0Error,
    Betti1Error,
    Dice,
    JaccardIndex,
    Precision,
    Recall,
    Specificity,
    clDice,
)
from src.registry import MODEL_REGISTRY as GLOBAL_MODEL_REGISTRY


class DiffusionModel(L.LightningModule):
    """Diffusion segmentation model with sliding window inference."""

    def __init__(
        self,
        arch_name: str = 'segdiff',
        image_size: int = 224,
        dim: int = 64,
        timesteps: int = 50,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
        num_ensemble: int = 1,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        soft_label_type: str = 'none',
        soft_label_cache: bool = True,
        soft_label_fg_max: int = 11,
        soft_label_thickness_max: int = 13,
        soft_label_kernel_ratio: float = 0.1,
        log_image_enabled: bool = False,
        log_image_names: list = None,
        loss_type: str = 'hybrid',
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name

        # Initialize soft label generator
        from src.data.transforms import SoftLabelGenerator
        self.soft_label_generator = SoftLabelGenerator(
            method=soft_label_type,
            cache=soft_label_cache,
            fg_max=soft_label_fg_max,
            thickness_max=soft_label_thickness_max,
            kernel_ratio=soft_label_kernel_ratio,
        )

        # Create diffusion model (registry 기반)
        if arch_name not in GLOBAL_MODEL_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(GLOBAL_MODEL_REGISTRY.keys())}")

        create_fn = GLOBAL_MODEL_REGISTRY.get(arch_name)
        self.diffusion_model = create_fn(
            image_size=image_size,
            dim=dim,
            timesteps=timesteps,
            loss_type=loss_type,
        )

        # EMA model for stable inference
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = deepcopy(self.diffusion_model)
            self.ema_model.eval()
            self.ema_model.requires_grad_(False)
            self.ema_decay = ema_decay
        else:
            self.ema_model = None

        # Sliding window inferer for validation
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )

        # Metrics
        self.val_metrics = MetricCollection({
            'dice': Dice(num_classes=num_classes, average='macro'),
            'precision': Precision(num_classes=num_classes, average='macro'),
            'recall': Recall(num_classes=num_classes, average='macro'),
            'specificity': Specificity(num_classes=num_classes, average='macro'),
            'iou': JaccardIndex(num_classes=num_classes, average='macro'),
        })

        # Test metrics (separate instance to avoid contamination)
        self.test_metrics = MetricCollection({
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

        # Test vessel metrics (separate instance)
        self.test_vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })

    def forward(self, img: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns loss during training.
        
        Args:
            img: Ground truth segmentation mask
            cond_img: Conditional image
        """
        return self.diffusion_model(img, cond_img)

    def update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.diffusion_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def sample(self, cond_img: torch.Tensor, save_steps: list = None) -> torch.Tensor:
        """Sample from diffusion model (inference).
        
        This function is called by sliding window inferer for each patch.
        Each patch goes through the full diffusion sampling process.
        
        Uses EMA model if available for more stable inference.
        
        Args:
            cond_img: Conditional image
            save_steps: List of timesteps to save for visualization
        """
        # Use EMA model for inference if available
        model = self.ema_model if (self.use_ema and self.ema_model is not None) else self.diffusion_model
        return model.sample(cond_img, save_steps)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Convert to float [0, 1] if needed
        if labels.dtype != torch.float32:
            labels = labels.float()
        if labels.max() > 1:
            labels = labels / 255.0

        # Generate soft labels as denoising target
        # Get sample IDs for caching (if available)
        sample_ids = batch.get('name', None)

        # Generate soft labels (returns binary labels if method='none')
        soft_labels = self.soft_label_generator(labels, sample_ids)

        # Use soft labels as x_0 target in diffusion forward process
        target_labels = soft_labels

        # Compute diffusion loss with soft labels as target
        loss = self(target_labels, images)

        # Log
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self.use_ema:
            self.update_ema()

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            pred_masks_list = []
            for _ in range(self.hparams.num_ensemble):
                pred_result = self.inferer(images, self.sample)
                # Handle dict output from v2 model
                if isinstance(pred_result, dict):
                    pred_masks = pred_result['mask']
                else:
                    pred_masks = pred_result
                pred_masks_list.append(pred_masks)
            # Average predictions
            pred_masks = torch.stack(pred_masks_list).mean(dim=0)
        else:
            # Single sampling
            pred_result = self.inferer(images, self.sample)
            # Handle dict output from v2 model
            if isinstance(pred_result, dict):
                pred_masks = pred_result['mask']
            else:
                pred_masks = pred_result

        # Convert predictions to class indices
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        preds = (pred_masks > 0.5).long()

        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)

        # Log
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)

        # Log images to TensorBoard (specified samples only)
        if hasattr(self.hparams, 'log_image_enabled') and self.hparams.log_image_enabled:
            log_names = getattr(self.hparams, 'log_image_names', None)
            pred_binary = (preds > 0).float()
            label_binary = (labels > 0).float()

            for i in range(images.shape[0]):
                sample_name = batch['name'][i] if 'name' in batch else f'sample_{i}'
                filename = sample_name.split('/')[-1] if '/' in sample_name else sample_name

                # Only log if filename matches log_image_names (or log all if not specified)
                if log_names is None or filename in log_names:
                    print(f"[DEBUG] Logging image: {filename} at epoch {self.current_epoch}")
                    # Log to TensorBoard
                    self.logger.experiment.add_image(
                        f'val/{filename}/prediction',
                        pred_binary[i:i+1],
                        self.global_step
                    )
                    self.logger.experiment.add_image(
                        f'val/{filename}/ground_truth',
                        label_binary[i:i+1],
                        self.global_step
                    )

        return general_metrics['dice']

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            pred_masks_list = []
            for _ in range(self.hparams.num_ensemble):
                pred_result = self.inferer(images, self.sample)
                # Handle dict output from v2 model
                if isinstance(pred_result, dict):
                    pred_masks = pred_result['mask']
                else:
                    pred_masks = pred_result
                pred_masks_list.append(pred_masks)
            # Average predictions
            pred_masks = torch.stack(pred_masks_list).mean(dim=0)
        else:
            # Single sampling
            pred_result = self.inferer(images, self.sample)
            # Handle dict output from v2 model
            if isinstance(pred_result, dict):
                pred_masks = pred_result['mask']
            else:
                pred_masks = pred_result

        # Convert predictions to class indices
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks.squeeze(1)
        preds = (pred_masks > 0.5).long()

        # Compute metrics (use test_metrics, not val_metrics!)
        general_metrics = self.test_metrics(preds, labels)
        vessel_metrics = self.test_vessel_metrics(preds, labels)

        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=20,
            factor=0.5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train/loss',
                'interval': 'epoch',
            }
        }
