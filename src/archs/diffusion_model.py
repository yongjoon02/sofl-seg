"""Diffusion models for vessel segmentation.
Based on supervised_model.py structure with MedSegDiff and BerDiff.
"""
from copy import deepcopy
import inspect
from pathlib import Path

import lightning.pytorch as L
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.segmentation.dice import DiceScore

from src.metrics import Betti0Error, Betti1Error, clDice
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
        conditioning_fmap_size: int | None = None,
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
        create_kwargs = {
            'image_size': image_size,
            'dim': dim,
            'timesteps': timesteps,
            'loss_type': loss_type,
        }
        if conditioning_fmap_size is not None:
            create_kwargs['conditioning_fmap_size'] = conditioning_fmap_size
        # Only pass supported args to factory
        sig = inspect.signature(create_fn)
        create_kwargs = {k: v for k, v in create_kwargs.items() if k in sig.parameters}
        self.diffusion_model = create_fn(**create_kwargs)

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
            'dice': DiceScore(num_classes=num_classes, average='macro', include_background=False, input_format="index"),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'specificity': BinarySpecificity(),
            'iou': BinaryJaccardIndex(),
        })

        # Test metrics (separate instance to avoid contamination)
        self.test_metrics = MetricCollection({
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

        # Test vessel metrics (separate instance)
        self.test_vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })

        self.eval_save_intermediate = False
        self.eval_intermediate_dir = None
        self.eval_intermediate_t = None
        self.eval_intermediate_steps = None
        self.eval_intermediate_max_samples = 0
        self._eval_intermediate_saved = set()

    def _parse_sample_name(self, sample_name: str) -> str:
        name = str(sample_name)
        if '/' in name:
            name = name.split('/')[-1]
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        return name

    def _select_intermediate_indices(self, sample_names: list) -> list:
        max_samples = int(getattr(self, 'eval_intermediate_max_samples', 0) or 0)
        if max_samples <= 0:
            return list(range(len(sample_names)))
        indices = []
        for i, name in enumerate(sample_names):
            key = self._parse_sample_name(name)
            if key in self._eval_intermediate_saved:
                continue
            indices.append(i)
            self._eval_intermediate_saved.add(key)
            if len(indices) >= max_samples:
                break
        return indices

    def _resolve_intermediate_steps(self, num_timesteps: int) -> list:
        steps = getattr(self, 'eval_intermediate_steps', None)
        if steps:
            return [int(s) for s in steps if 0 <= int(s) < num_timesteps]
        t_list = getattr(self, 'eval_intermediate_t', None)
        if not t_list:
            return []
        resolved = []
        for t in t_list:
            t_val = max(0.0, min(1.0, float(t)))
            idx = int(round(t_val * (num_timesteps - 1)))
            resolved.append(idx)
        return sorted(set(resolved))

    def _save_intermediate_images(
        self,
        sample_names: list,
        steps_dict: dict,
        out_dir: str,
        labels: list | None,
    ) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # 추가: binary mask 저장용 폴더
        out_path_binary = out_path.parent / (out_path.name + "_binary")
        out_path_binary.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Saving intermediate images to: {out_path}", flush=True)
        print(f"[DEBUG] Saving binary intermediate images to: {out_path_binary}", flush=True)

        if not steps_dict:
            print(f"[DEBUG] steps_dict is empty! No intermediate steps to save.", flush=True)
        else:
            print(f"[DEBUG] steps_dict keys: {list(steps_dict.keys())}", flush=True)
        ordered_steps = sorted(steps_dict.keys())
        for i, name in enumerate(sample_names):
            sample_dir = out_path / self._parse_sample_name(name)
            sample_dir.mkdir(parents=True, exist_ok=True)
            # binary 저장용
            sample_dir_binary = out_path_binary / self._parse_sample_name(name)
            sample_dir_binary.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] Sample {i}: {name} -> {sample_dir}, binary: {sample_dir_binary}", flush=True)

            row_imgs = []
            row_imgs_binary = []
            per_tile_labels = []
            for idx, step_idx in enumerate(ordered_steps):
                step_tensor = steps_dict.get(step_idx)
                if not torch.is_tensor(step_tensor):
                    print(f"[DEBUG] step_tensor for step {step_idx} is not a tensor!", flush=True)
                    continue
                img = step_tensor[i].detach().float().cpu()
                if img.dim() == 3:
                    img = img.squeeze(0)
                img = img.clamp(0.0, 1.0)
                # soft mask 저장
                img_soft = (img.numpy() * 255.0).astype(np.uint8)
                img_soft = img_soft.T
                row_imgs.append(img_soft)
                # binary mask 저장
                img_binary = (img > 0.5).numpy().astype(np.uint8) * 255
                img_binary = img_binary.T
                row_imgs_binary.append(img_binary)
                # 개별 step 저장 (binary)
                pil_binary = Image.fromarray(img_binary, mode='L')
                binary_path = sample_dir_binary / f"step_{step_idx}.png"
                try:
                    pil_binary.save(binary_path)
                    print(f"[DEBUG] Saved binary mask: {binary_path}", flush=True)
                except Exception as e:
                    print(f"[ERROR] Failed to save binary mask: {binary_path}, error: {e}", flush=True)

                if labels is not None and idx < len(labels):
                    per_tile_labels.append(labels[idx])
                else:
                    per_tile_labels.append(f"step={int(step_idx)}")
            # soft mask row 저장 (기존)
            if row_imgs:
                row = np.concatenate(row_imgs, axis=1)
                tile_h, tile_w = row_imgs[0].shape
                font_size = max(12, int(tile_h * 0.08))
                font = None
                for path in (
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                ):
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except Exception:
                        continue
                if font is None:
                    font = ImageFont.load_default()
                tmp = Image.new("L", (1, 1), color=255)
                draw_tmp = ImageDraw.Draw(tmp)
                text_heights = [draw_tmp.textbbox((0, 0), label, font=font)[3] for label in per_tile_labels]
                top_margin = (max(text_heights) if text_heights else font_size) + 6
                canvas = Image.new("L", (row.shape[1], row.shape[0] + top_margin), color=255)
                canvas.paste(Image.fromarray(row, mode="L"), (0, top_margin))
                draw = ImageDraw.Draw(canvas)
                for idx, label in enumerate(per_tile_labels):
                    x0 = idx * tile_w
                    x1 = x0 + tile_w
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    x = x0 + (tile_w - text_w) // 2
                    y = 2
                    draw.text((x, y), label, fill=0, font=font)
                canvas.save(sample_dir / "steps_row.png")
            # binary mask row 저장
            if row_imgs_binary:
                row_binary = np.concatenate(row_imgs_binary, axis=1)
                canvas_binary = Image.new("L", (row_binary.shape[1], row_binary.shape[0] + top_margin), color=255)
                canvas_binary.paste(Image.fromarray(row_binary, mode="L"), (0, top_margin))
                draw_binary = ImageDraw.Draw(canvas_binary)
                for idx, label in enumerate(per_tile_labels):
                    x0 = idx * tile_w
                    x1 = x0 + tile_w
                    bbox = draw_binary.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    x = x0 + (tile_w - text_w) // 2
                    y = 2
                    draw_binary.text((x, y), label, fill=0, font=font)
                canvas_binary.save(sample_dir_binary / "steps_row.png")

    def _maybe_save_intermediate_eval(self, images: torch.Tensor, sample_names: list) -> None:
        if not getattr(self, 'eval_save_intermediate', False):
            return
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'is_global_zero') and not self.trainer.is_global_zero:
            return
        out_dir = getattr(self, 'eval_intermediate_dir', None)
        if not out_dir:
            return
        indices = self._select_intermediate_indices(sample_names)
        if not indices:
            return
        model = self.ema_model if (self.use_ema and self.ema_model is not None) else self.diffusion_model
        num_timesteps = int(getattr(model, 'num_timesteps', 0))
        if num_timesteps <= 0:
            return
        save_steps = self._resolve_intermediate_steps(num_timesteps)
        if not save_steps:
            return
        label_list = None
        if self.eval_intermediate_t:
            label_list = []
            idx_map = {}
            for t in self.eval_intermediate_t:
                t_val = max(0.0, min(1.0, float(t)))
                idx = int(round(t_val * (num_timesteps - 1)))
                if idx not in idx_map:
                    idx_map[idx] = f"t={t_val:.2f}"
            for step_idx in sorted(save_steps):
                label_list.append(idx_map.get(step_idx, f"step={int(step_idx)}"))
        subset_images = images[indices]
        subset_names = [sample_names[i] for i in indices]
        sample_out = model.sample(subset_images, save_steps=save_steps)
        steps_dict = sample_out.get('steps', {}) if isinstance(sample_out, dict) else {}
        if steps_dict:
            self._save_intermediate_images(subset_names, steps_dict, out_dir, label_list)

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

        # Get sample names if available (fix for NameError)
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
