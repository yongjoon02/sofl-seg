"""Supervised learning model for OCT segmentation."""

import inspect
import warnings

import lightning.pytorch as L
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection

from src.losses import SoftBCELoss, SoftCrossEntropyLoss, TopoLoss, L1Loss, L2Loss, HuberLoss
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
from src.registry import LOSS_REGISTRY, MODEL_REGISTRY as GLOBAL_MODEL_REGISTRY


class ModelWrapper(nn.Module):
    """Wrapper to handle models that return dict outputs (e.g., UNet3Plus)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # If model returns dict, extract main output
        if isinstance(output, dict):
            return output['main_out']
        return output


class SupervisedModel(L.LightningModule):
    """Supervised segmentation model with sliding window inference."""

    def __init__(
        self,
        arch_name: str = 'cenet',
        in_channels: int = 1,
        num_classes: int = 2,
        img_size: int = 224,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
        experiment_name: str = None,
        data_name: str = 'octa500_3m',
        log_image_enabled: bool = False,
        log_image_names: list = None,
        soft_label: bool = False,  # soft label 학습 모드
        loss_type: str = 'ce',  # 'ce' / 'bce' / 'l1' / 'l2' / 'bce_l1' / 'bce_l2' / 'bce_huber' / 'bce_topo'
        l1_lambda: float = 0.2,  # Weight for L1 loss when using bce_l1
        l2_lambda: float = 0.5,  # Weight for L2 loss when using bce_l2
        huber_lambda: float = 0.2,  # Weight for Huber loss when using bce_huber
        topo_lambda: float = 0.1,
        topo_size: int = 100,
        topo_pers_thresh: float = 0.0,
        topo_pers_thresh_perfect: float = 0.99,
        loss_name: str = None,  # Deprecated in favor of loss config
        loss_kwargs: dict = None,
        loss: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name
        self.arch_name = arch_name
        self.soft_label = soft_label  # Store soft label mode

        # 모델 생성 (registry 기반)
        base_model = self._build_model(arch_name, in_channels, num_classes)

        # Wrap model to handle dict outputs
        self.model = ModelWrapper(base_model)

        # Sliding window inferer for validation uses configured img_size
        self.inferer = SlidingWindowInferer(
            roi_size=(img_size, img_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )

        # Loss function (registry 우선, 없으면 기존 분기)
        self.loss_fn = self._build_loss(
            loss_cfg=loss,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs or {},
            loss_type=loss_type,
            soft_label=soft_label,
            l1_lambda=l1_lambda,
            l2_lambda=l2_lambda,
            huber_lambda=huber_lambda,
            topo_lambda=topo_lambda,
            topo_size=topo_size,
            topo_pers_thresh=topo_pers_thresh,
            topo_pers_thresh_perfect=topo_pers_thresh_perfect,
        )

        # Optional: allow custom losses to use hard labels as an additional signal
        # (e.g., soft regression target + Dice regularizer to hard GT).
        self._loss_accepts_hard_labels = False
        if not isinstance(self.loss_fn, nn.ModuleDict):
            try:
                sig = inspect.signature(self.loss_fn.forward)
                self._loss_accepts_hard_labels = 'hard_labels' in sig.parameters
            except (TypeError, ValueError):
                self._loss_accepts_hard_labels = False

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Get geometry if available (soft label case)
        geometry = batch.get('geometry', None)

        # Squeeze channel dimension if present
        if labels.dim() == 4:
            labels = labels.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        if geometry is not None and geometry.dim() == 4:
            geometry = geometry.squeeze(1)
        
        # Choose training target based on soft_label mode
        if self.soft_label and geometry is not None:
            # Soft label 학습: geometry (soft label) 사용
            train_labels = geometry
        else:
            # Hard label 학습: labels 사용
            train_labels = labels
            if not self.soft_label:
                train_labels = train_labels.long()

        # Forward (ModelWrapper handles dict outputs)
        logits = self(images)
        
        # Check for NaN/Inf in logits with detailed info
        nan_mask = torch.isnan(logits)
        inf_mask = torch.isinf(logits)
        if nan_mask.any() or inf_mask.any():
            nan_count = nan_mask.sum().item()
            inf_count = inf_mask.sum().item()
            total_elements = logits.numel()
            print(f"⚠️ WARNING: Logits contain NaN/Inf at step {batch_idx}")
            print(f"   NaN: {nan_count}/{total_elements} ({100*nan_count/total_elements:.2f}%)")
            print(f"   Inf: {inf_count}/{total_elements} ({100*inf_count/total_elements:.2f}%)")
            print(f"   Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            
            # Check if images have issues
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"   ⚠️ Images also contain NaN/Inf!")
            
            # Replace NaN/Inf with zeros to prevent crash
            logits = torch.where(nan_mask | inf_mask, 
                                torch.zeros_like(logits), logits)

        # Compute loss
        if isinstance(self.loss_fn, nn.ModuleDict):
            bce_loss = self.loss_fn['bce'](logits, train_labels)
            
            # Check each loss individually
            if torch.isnan(bce_loss) or torch.isinf(bce_loss):
                print(f"⚠️ WARNING: BCE loss is NaN/Inf at step {batch_idx}: {bce_loss}")
                bce_loss = torch.tensor(0.0, device=bce_loss.device, dtype=bce_loss.dtype, requires_grad=True)
            
            # Handle different loss combinations
            if 'topo' in self.loss_fn:
                # BCE + TopoLoss
                topo_loss = self.loss_fn['topo'](logits, train_labels)
                
                if torch.isnan(topo_loss) or torch.isinf(topo_loss):
                    print(f"⚠️ WARNING: TopoLoss is NaN/Inf at step {batch_idx}: {topo_loss}")
                    topo_loss = torch.tensor(0.0, device=topo_loss.device, dtype=topo_loss.dtype, requires_grad=True)
                
                # Normalize TopoLoss by BCE loss scale for better balance
                if bce_loss.item() > 0:
                    topo_loss_normalized = topo_loss / (bce_loss.item() + 1e-8) * bce_loss
                else:
                    topo_loss_normalized = topo_loss
                
                loss = bce_loss + topo_loss_normalized
                
                # Log individual losses
                self.log('train/bce_loss', bce_loss, prog_bar=False)
                self.log('train/topo_loss', topo_loss, prog_bar=False)
                self.log('train/topo_loss_normalized', topo_loss_normalized, prog_bar=False)
            
            elif 'l2' in self.loss_fn:
                # BCE + L2 loss
                l2_loss = self.loss_fn['l2'](logits, train_labels)
                
                if torch.isnan(l2_loss) or torch.isinf(l2_loss):
                    print(f"⚠️ WARNING: L2 loss is NaN/Inf at step {batch_idx}: {l2_loss}")
                    l2_loss = torch.tensor(0.0, device=l2_loss.device, dtype=l2_loss.dtype, requires_grad=True)
                
                # Combine BCE and L2 with weight
                loss = bce_loss + self.l2_lambda * l2_loss
                
                # Log individual losses
                self.log('train/bce_loss', bce_loss, prog_bar=False)
                self.log('train/l2_loss', l2_loss, prog_bar=False)
            
            elif 'huber' in self.loss_fn:
                # BCE + Huber loss
                huber_loss = self.loss_fn['huber'](logits, train_labels)
                
                if torch.isnan(huber_loss) or torch.isinf(huber_loss):
                    print(f"⚠️ WARNING: Huber loss is NaN/Inf at step {batch_idx}: {huber_loss}")
                    huber_loss = torch.tensor(0.0, device=huber_loss.device, dtype=huber_loss.dtype, requires_grad=True)
                
                # Combine BCE and Huber with weight
                loss = bce_loss + self.huber_lambda * huber_loss
                
                # Log individual losses
                self.log('train/bce_loss', bce_loss, prog_bar=False)
                self.log('train/huber_loss', huber_loss, prog_bar=False)
            
            elif 'l1' in self.loss_fn:
                # BCE + L1 loss
                l1_loss = self.loss_fn['l1'](logits, train_labels)
                
                if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                    print(f"⚠️ WARNING: L1 loss is NaN/Inf at step {batch_idx}: {l1_loss}")
                    l1_loss = torch.tensor(0.0, device=l1_loss.device, dtype=l1_loss.dtype, requires_grad=True)
                
                # Combine BCE and L1 with weight
                loss = bce_loss + self.l1_lambda * l1_loss
                
                # Log individual losses
                self.log('train/bce_loss', bce_loss, prog_bar=False)
                self.log('train/l1_loss', l1_loss, prog_bar=False)
            
            else:
                # Fallback: just use BCE
                loss = bce_loss
                self.log('train/bce_loss', bce_loss, prog_bar=False)
        else:
            if self._loss_accepts_hard_labels:
                out = self.loss_fn(logits, train_labels, hard_labels=labels)
            else:
                out = self.loss_fn(logits, train_labels)

            if isinstance(out, tuple) and len(out) == 2:
                loss, loss_dict = out
                if isinstance(loss_dict, dict):
                    for name, value in loss_dict.items():
                        self.log(f"train/{name}_loss", value, prog_bar=False)
            else:
                loss = out
            
            # Check final loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ WARNING: Total loss is NaN/Inf at step {batch_idx}: {loss}")
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)

        # Final check before logging
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ ERROR: Loss is still NaN/Inf after fixes at step {batch_idx}")
            # Return a small dummy loss to prevent training crash
            loss = torch.tensor(1e-6, device=loss.device, dtype=loss.dtype, requires_grad=True)

        # Log
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        
        # Get geometry if available (soft label case)
        geometry = batch.get('geometry', None)
        if geometry is not None and geometry.dim() == 4:
            geometry = geometry.squeeze(1)

        # Convert soft/binary labels to class indices (threshold for soft labels)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels_binary = (labels > 0.5).long()

        # Sliding window inference (ModelWrapper handles dict outputs)
        logits = self.inferer(images, self.model)

        # Compute loss (validation에서는 TopoLoss 제외: DDP 안전 및 속도)
        # Loss 계산 방식:
        # - Training: soft label과 비교 (학습 목표와 일치)
        # - Validation: soft label과 비교 (일관성) 또는 hard label과 비교 (최종 성능 예측)
        # 여기서는 학습 목표와 일치하도록 soft label 사용
        # Metrics는 항상 hard label로 계산 (표준 평가 방법)
        if self.soft_label and geometry is not None:
            # Soft label 학습 시: geometry (soft label)과 비교
            loss_labels = geometry
        else:
            # Hard label 학습 시: labels와 비교
            loss_labels = labels
        
        # Compute loss (validation에서는 Topo 제외, 나머지는 조합 유지)
        if isinstance(self.loss_fn, nn.ModuleDict):
            bce_loss = self.loss_fn['bce'](logits, loss_labels)
            if 'l2' in self.loss_fn:
                l2_loss = self.loss_fn['l2'](logits, loss_labels)
                loss = bce_loss + self.l2_lambda * l2_loss
            elif 'l1' in self.loss_fn:
                l1_loss = self.loss_fn['l1'](logits, loss_labels)
                loss = bce_loss + self.l1_lambda * l1_loss
            elif 'huber' in self.loss_fn:
                huber_loss = self.loss_fn['huber'](logits, loss_labels)
                loss = bce_loss + self.huber_lambda * huber_loss
            else:
                loss = bce_loss
        else:
            if self._loss_accepts_hard_labels:
                out = self.loss_fn(logits, loss_labels, hard_labels=labels)
            else:
                out = self.loss_fn(logits, loss_labels)

            if isinstance(out, tuple) and len(out) == 2:
                loss, loss_dict = out
                if isinstance(loss_dict, dict):
                    for name, value in loss_dict.items():
                        self.log(f"val/{name}_loss", value, prog_bar=False)
            else:
                loss = out

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        
        # Get output geometry (softmax probability for foreground class)
        output_geometry = torch.softmax(logits, dim=1)[:, 1, :, :]  # (B, H, W)

        # General metrics
        general_metrics = self.val_metrics(preds, labels_binary)

        # Vessel-specific metrics
        vessel_metrics = self.vessel_metrics(preds, labels_binary)

        # Log
        self.log('val/loss', loss, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False)

        # Log images to TensorBoard (specified samples only)
        if hasattr(self.hparams, 'log_image_enabled') and self.hparams.log_image_enabled:
            log_names = getattr(self.hparams, 'log_image_names', None)
            pred_binary = (preds > 0).float()
            label_binary = (labels_binary > 0).float()

            for i in range(images.shape[0]):
                sample_name = batch['name'][i] if 'name' in batch else f'sample_{i}'
                filename = sample_name.split('/')[-1] if '/' in sample_name else sample_name

                # Only log if filename matches log_image_names (or log all if not specified)
                if log_names is None or filename in log_names:
                    print(f"[DEBUG] Logging image: {filename} at epoch {self.current_epoch}")
                    
                    # Normalize image for visualization
                    img = (images[i] + 1) / 2 if images[i].min() < 0 else images[i]
                    if img.dim() == 2:
                        img = img.unsqueeze(0)
                    
                    # Standard visualization: [image, label, pred]
                    vis_row = torch.cat([img, label_binary[i:i+1], pred_binary[i:i+1]], dim=-1)
                    self.logger.experiment.add_image(
                        f'val/{filename}',
                        vis_row,
                        self.global_step
                    )
                    
                    # If geometry is provided (soft label), also log geometry comparison
                    if geometry is not None and output_geometry is not None:
                        geom = geometry[i].float().unsqueeze(0)
                        out_geom = output_geometry[i].float().unsqueeze(0)
                        
                        # Debug: Check if geometry is actually soft label
                        # Note: After cropping, some crops might appear binary if they only contain
                        # background or foreground regions. This is normal and not a problem.
                        geom_min, geom_max = geom.min().item(), geom.max().item()
                        geom_unique = torch.unique(geom).numel()
                        if geom_unique <= 2 and geom_min in [0.0, 1.0] and geom_max in [0.0, 1.0]:
                            # This can happen if the crop only contains background or foreground
                            # It's not necessarily a problem - just a small crop region
                            pass  # Don't warn - this is expected for some crops
                        else:
                            # Only log if it's actually a soft label (to reduce log spam)
                            if not hasattr(self, '_geometry_logged') or filename not in getattr(self, '_geometry_logged', set()):
                                print(f"✅ geometry for {filename} is soft label (unique values: {geom_unique}, range: [{geom_min:.3f}, {geom_max:.3f}])")
                                if not hasattr(self, '_geometry_logged'):
                                    self._geometry_logged = set()
                                self._geometry_logged.add(filename)
                        
                        # Geometry visualization: [image, target_geometry, output_geometry]
                        geom_vis_row = torch.cat([img, geom, out_geom], dim=-1)
                        self.logger.experiment.add_image(
                            f'val_geometry/{filename}',
                            geom_vis_row,
                            self.global_step
                        )

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        # Convert soft/binary labels to class indices (threshold for soft labels)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        # Sliding window inference (ModelWrapper handles dict outputs)
        logits = self.inferer(images, self.model)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)

        # General metrics (use test_metrics, not val_metrics!)
        general_metrics = self.test_metrics(preds, labels)

        # Vessel-specific metrics (use test_vessel_metrics!)
        vessel_metrics = self.test_vessel_metrics(preds, labels)

        # Log
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()})
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()})


        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            # Convert to binary masks for visualization
            pred_masks = (preds > 0).float()
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
                sample_names, images, pred_masks, label_masks, sample_metrics
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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _build_model(self, arch_name: str, in_channels: int, num_classes: int) -> nn.Module:
        """Fetch model class from global registry and instantiate."""
        if arch_name not in GLOBAL_MODEL_REGISTRY:
            available = list(GLOBAL_MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown architecture: {arch_name}. Choose from {available}")

        model_cls = GLOBAL_MODEL_REGISTRY.get(arch_name)
        return model_cls(in_channels=in_channels, num_classes=num_classes)

    def _build_loss(
        self,
        *,
        loss_cfg: dict | None,
        loss_name: str | None,
        loss_kwargs: dict,
        loss_type: str,
        soft_label: bool,
        l1_lambda: float,
        l2_lambda: float,
        huber_lambda: float,
        topo_lambda: float,
        topo_size: int,
        topo_pers_thresh: float,
        topo_pers_thresh_perfect: float,
    ):
        """Construct loss. Registry 우선, 없으면 기존 분기 사용."""
        # Ensure built-in losses are imported and registered in LOSS_REGISTRY.
        import src.losses  # noqa: F401

        if loss_cfg:
            cfg_name = loss_cfg.get('name')
            cfg_params = loss_cfg.get('params', {})
            if cfg_name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss: {cfg_name}. Available: {LOSS_REGISTRY.keys()}")
            loss_cls = LOSS_REGISTRY.get(cfg_name)
            sig = inspect.signature(loss_cls.__init__)
            if 'soft_label' in sig.parameters and 'soft_label' not in cfg_params:
                cfg_params = {**cfg_params, 'soft_label': soft_label}
            return loss_cls(**cfg_params)

        if loss_name:
            if loss_name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss: {loss_name}. Available: {LOSS_REGISTRY.keys()}")
            loss_cls = LOSS_REGISTRY.get(loss_name)

            # soft_label 지원 인자가 있으면 자동 주입
            sig = inspect.signature(loss_cls.__init__)
            if 'soft_label' in sig.parameters and 'soft_label' not in loss_kwargs:
                loss_kwargs = {**loss_kwargs, 'soft_label': soft_label}
            return loss_cls(**loss_kwargs)

        # 기존 하드코딩 분기 (backward compatibility)
        if loss_type == 'bce':
            return SoftBCELoss(soft_label=soft_label)
        if loss_type == 'l1':
            return L1Loss(soft_label=soft_label)
        if loss_type == 'l2':
            return L2Loss(soft_label=soft_label)
        if loss_type == 'bce_l1':
            self.l1_lambda = l1_lambda
            return nn.ModuleDict({
                'bce': SoftBCELoss(soft_label=soft_label),
                'l1': L1Loss(soft_label=soft_label),
            })
        if loss_type == 'bce_l2':
            self.l2_lambda = l2_lambda
            return nn.ModuleDict({
                'bce': SoftBCELoss(soft_label=soft_label),
                'l2': L2Loss(soft_label=soft_label),
            })
        if loss_type == 'bce_huber':
            self.huber_lambda = huber_lambda
            return nn.ModuleDict({
                'bce': SoftBCELoss(soft_label=soft_label),
                'huber': HuberLoss(soft_label=soft_label),
            })
        if loss_type == 'bce_topo':
            if soft_label:
                warnings.warn(
                    "TopoLoss is disabled when soft_label=True. "
                    "Topology loss requires binary hard masks, not soft labels. "
                    "Using BCE only."
                )
                return SoftBCELoss(soft_label=soft_label)
            return nn.ModuleDict({
                'bce': SoftBCELoss(soft_label=False),
                'topo': TopoLoss(
                    lambda_weight=topo_lambda,
                    topo_size=topo_size,
                    pers_thresh=topo_pers_thresh,
                    pers_thresh_perfect=topo_pers_thresh_perfect,
                )
            })

        # default: cross entropy
        return SoftCrossEntropyLoss(soft_label=soft_label)
