"""Training runner - config-based training logic."""

import inspect
import os
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
    Callback,
)
from lightning.pytorch.tuner import Tuner

from src.archs.diffusion_model import DiffusionModel
from src.archs.supervised_model import SupervisedModel
from src.experiment import EnhancedTensorBoardLogger, ExperimentTracker
from src.registry import get_dataset_info, get_model_info
from src.utils.config import save_config

# Import data modules to register datasets
import src.data  # noqa: F401


class EpochSummaryLogger(Callback):
    """Print compact epoch summaries (rank 0 only)."""

    def __init__(self, max_epochs: int):
        super().__init__()
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            cur = trainer.current_epoch
            print(f"[Epoch {cur}/{self.max_epochs}] start")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            cur = trainer.current_epoch
            metrics = trainer.callback_metrics
            # Extract a few common metrics if present
            train_loss = metrics.get('train/loss')
            val_dice = metrics.get('val/dice')
            val_loss = metrics.get('val/loss')
            parts = [f"[Epoch {cur}/{self.max_epochs}] done"]
            if train_loss is not None:
                parts.append(f"train/loss={float(train_loss):.4f}")
            if val_loss is not None:
                parts.append(f"val/loss={float(val_loss):.4f}")
            if val_dice is not None:
                parts.append(f"val/dice={float(val_dice):.4f}")
            print(" | ".join(parts))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0) in DDP."""
    # Check Lightning's LOCAL_RANK
    local_rank = os.environ.get('LOCAL_RANK', '0')
    return local_rank == '0'


class TrainRunner:
    """Config-based training runner."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize from config dictionary.
        
        Args:
            config: Full configuration (model, data, trainer sections)
        """
        # Extract config sections
        model_cfg = config['model']
        data_cfg = config['data']
        trainer_cfg = config['trainer']

        self.model_name = model_cfg['arch_name']
        self.dataset_name = data_cfg['name']

        # Get registry info
        self.model_info = get_model_info(self.model_name)
        self.dataset_info = get_dataset_info(self.dataset_name)

        # Store config
        self.config = config
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.trainer_cfg = trainer_cfg

        # Extract special flags
        self.tag = config.get('tag', None)
        self.resume = config.get('resume', None)
        self.debug = config.get('debug', False)
        self.lr_find_mode = self.trainer_cfg.get('lr_find', False)

        # Create experiment tracker (only on main process)
        self.tracker = ExperimentTracker() if is_main_process() else None

    def run(self):
        """Execute training."""
        if self.lr_find_mode:
            return self._run_lr_find()

        # Only print info on main process
        if is_main_process():
            self._print_info()

        # Create experiment only on main process
        if is_main_process():
            experiment = self.tracker.create_experiment(
                model=self.model_name,
                dataset=self.dataset_name,
                config=self.config,
                tag=self.tag,
            )
            self._experiment_id = experiment.id
            self._experiment_dir = experiment.dir

            print(f"\n{'='*60}")
            print(f"Experiment ID: {experiment.id}")
            print(f"Experiment dir: {experiment.dir}")
            print(f"{'='*60}\n")

            # Save config to experiment dir
            save_config(self.config, experiment.dir / "config.yaml")
        else:
            # Non-main processes wait and use a dummy dir
            # In DDP, all processes need to have the same experiment dir for checkpoints
            import time
            time.sleep(2)  # Wait for main process to create experiment
            # Use a temporary dir that won't be used
            self._experiment_dir = Path("/tmp/ddp_worker")
            self._experiment_dir.mkdir(parents=True, exist_ok=True)
            self._experiment_id = None

        try:
            # Create components
            datamodule = self._create_datamodule()
            model = self._create_model()
            callbacks = self._create_callbacks(self._experiment_dir)
            logger = self._create_logger(self._experiment_dir, self._experiment_id) if is_main_process() else None
            trainer = self._create_trainer(callbacks, logger)

            # Train
            if is_main_process():
                print("Starting training...")
            # PyTorch 2.6+ defaults torch.load(weights_only=True); resume checkpoints
            # often include non-weight objects. Allow full load when resuming.
            weights_only = self.trainer_cfg.get('weights_only', None)
            if weights_only is None and self.resume:
                weights_only = False
            trainer.fit(model, datamodule, ckpt_path=self.resume, weights_only=weights_only)

            # Finish experiment (only on main process)
            if is_main_process():
                final_metrics = {
                    k: v.item() if torch.is_tensor(v) else v
                    for k, v in trainer.callback_metrics.items()
                    if 'val/' in k
                }
                best_ckpt = self._experiment_dir / "checkpoints" / "best.ckpt"

                self.tracker.finish_experiment(
                    self._experiment_id,
                    final_metrics=final_metrics,
                    best_checkpoint=str(best_ckpt) if best_ckpt.exists() else None,
                )

                print(f"\n{'='*60}")
                print("✅ Training completed!")
                print(f"   Best checkpoint: {best_ckpt}")
                print("   Final metrics:")
                for k, v in final_metrics.items():
                    print(f"      {k}: {v:.4f}")
                print(f"{'='*60}\n")

        except KeyboardInterrupt:
            if is_main_process():
                print("\n⚠️  Training interrupted by user")
                self.tracker.mark_failed(self._experiment_id, "Interrupted by user")
            raise
        except Exception as e:
            if is_main_process():
                print(f"\n❌ Training failed: {e}")
                self.tracker.mark_failed(self._experiment_id, str(e))
            raise

    def _run_lr_find(self):
        """Run Lightning LR finder and report suggested LR."""
        if is_main_process():
            print("\n🔍 Running LR finder (no training will be performed)...")

        # Minimal setup (no experiment tracking/logging)
        datamodule = self._create_datamodule()
        model = self._create_model()

        # Use temp dir for callbacks/checkpoints
        tmp_dir = Path("lr_find_runs")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        callbacks = self._create_callbacks(tmp_dir)
        trainer = self._create_trainer(callbacks, logger=None)

        tuner = Tuner(trainer)
        lr_find_kwargs = {}
        min_lr = self.trainer_cfg.get('lr_find_min_lr')
        max_lr = self.trainer_cfg.get('lr_find_max_lr')
        if min_lr is not None:
            lr_find_kwargs['min_lr'] = min_lr
        if max_lr is not None:
            lr_find_kwargs['max_lr'] = max_lr
        lr_finder = tuner.lr_find(model, datamodule=datamodule, **lr_find_kwargs)
        suggested = lr_finder.suggestion()

        if is_main_process():
            print(f"\n✅ LR finder finished. Suggested LR: {suggested:.6f}")
            print("Plot is available via lr_finder.plot(show=True/False) if needed.")
        return suggested

    def _print_info(self):
        """Print training info."""
        print(f"\n{'='*60}")
        print("Training Configuration")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"  - Task: {self.model_info.task}")
        print(f"  - Parameters: {self.model_info.params:,}")
        print(f"\nDataset: {self.dataset_name}")
        print(f"  - Samples: {self.dataset_info.num_train} train / "
              f"{self.dataset_info.num_val} val / {self.dataset_info.num_test} test")
        print("\nHyperparameters:")
        print(f"  - Learning rate: {self.model_cfg.get('learning_rate')}")
        print(f"  - Batch size: {self.data_cfg.get('train_bs')}")
        print(f"  - Crop size: {self.data_cfg.get('crop_size')}")
        print(f"  - Max epochs: {self.trainer_cfg.get('max_epochs')}")
        if self.model_info.task == 'diffusion':
            print(f"  - Timesteps: {self.model_cfg.get('timesteps')}")
            print(f"  - Soft label: {self.model_cfg.get('soft_label_type')}")
            print(f"  - Use EMA: {self.model_cfg.get('use_ema')}")
        print(f"{'='*60}\n")

    def _create_datamodule(self):
        """Create datamodule."""
        DataModuleClass = self.dataset_info.class_ref
        params = set(inspect.signature(DataModuleClass.__init__).parameters)

        # Build kwargs based on dataset
        kwargs = {
            'train_dir': self.data_cfg.get('train_dir', self.dataset_info.default_train_dir),
            'val_dir': self.data_cfg.get('val_dir', self.dataset_info.default_val_dir),
            'test_dir': self.data_cfg.get('test_dir', self.dataset_info.default_test_dir),
            'crop_size': self.data_cfg.get('crop_size', self.dataset_info.default_crop_size),
            'train_bs': self.data_cfg.get('train_bs', 8),
        }
        
        # Add optional parameters if specified in config
        if 'num_samples_per_image' in self.data_cfg:
            kwargs['num_samples_per_image'] = self.data_cfg['num_samples_per_image']
        if 'label_subdir' in self.data_cfg:
            kwargs['label_subdir'] = self.data_cfg['label_subdir']
        if 'use_sauna_transform' in self.data_cfg:
            kwargs['use_sauna_transform'] = self.data_cfg['use_sauna_transform']
        if 'train_manual_dir' in self.data_cfg:
            kwargs['train_manual_dir'] = self.data_cfg['train_manual_dir']
        if 'train_sam_dir' in self.data_cfg:
            kwargs['train_sam_dir'] = self.data_cfg['train_sam_dir']
        if 'use_sam' in self.data_cfg:
            kwargs['use_sam'] = self.data_cfg['use_sam']

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        return DataModuleClass(**filtered_kwargs)

    def _create_model(self):
        """Create model."""
        common_args = {
            'arch_name': self.model_name,
            'learning_rate': self.model_cfg.get('learning_rate', self.model_info.default_lr),
            'weight_decay': self.model_cfg.get('weight_decay', 1e-5),
            # Prefer explicit config values over inferred defaults.
            'experiment_name': self.model_cfg.get('experiment_name', f"{self.dataset_name}/{self.model_name}"),
            'data_name': self.model_cfg.get('data_name', self.dataset_name),
            'image_size': self.model_cfg.get('image_size', 224),
            'num_classes': self.model_cfg.get('num_classes', 2),
        }

        # FlowModel 지원: registry task가 flow이면 FlowModel 사용
        if self.model_info.task == 'flow':
            from src.archs.flow_model import FlowModel
            return FlowModel(
                **common_args,
                patch_plan=self.model_cfg.get('patch_plan', None),
                dim=self.model_cfg.get('dim', 32),
                timesteps=self.model_cfg.get('timesteps', 15),
                sigma=self.model_cfg.get('sigma', 0.25),
                num_ensemble=self.model_cfg.get('num_ensemble', 1),
                log_image_enabled=self.model_cfg.get('log_image_enabled', False),
                log_image_names=self.model_cfg.get('log_image_names', None),
                model_channels=self.model_cfg.get('model_channels', 32),
                channel_mult=self.model_cfg.get('channel_mult', [1,2,4,8]),
                channel_mult_emb=self.model_cfg.get('channel_mult_emb', 4),
                num_blocks=self.model_cfg.get('num_blocks', 3),
                attn_resolutions=self.model_cfg.get('attn_resolutions', [16,16,8,8]),
                dropout=self.model_cfg.get('dropout', 0.0),
                label_dim=self.model_cfg.get('label_dim', 0),
                augment_dim=self.model_cfg.get('augment_dim', 0),
                # Loss configuration
                loss_type=self.model_cfg.get('loss_type', 'l2'),
                bce_weight=self.model_cfg.get('bce_weight', 0.5),
                l2_weight=self.model_cfg.get('l2_weight', 0.1),
                dice_weight=self.model_cfg.get('dice_weight', 0.2),
                loss=self.model_cfg.get('loss', None),
            )
        elif self.model_info.task == 'supervised':
            # SupervisedModel uses img_size instead of image_size
            supervised_args = common_args.copy()
            supervised_args['img_size'] = supervised_args.pop('image_size')
            supervised_args['in_channels'] = 1
            supervised_args['log_image_enabled'] = self.model_cfg.get('log_image_enabled', False)
            supervised_args['log_image_names'] = self.model_cfg.get('log_image_names', None)
            
            # Soft label support: prefer explicit model.soft_label, else infer from label_subdir.
            if 'soft_label' in self.model_cfg:
                supervised_args['soft_label'] = bool(self.model_cfg.get('soft_label'))
            else:
                label_subdir = self.data_cfg.get('label_subdir', 'label')
                supervised_args['soft_label'] = (label_subdir != 'label')
            
            # Loss type: 'ce' (default) or registry name.
            # For convenience, if loss_type matches a registered loss and no explicit
            # `model.loss` block exists, treat it as registry-based loss config.
            loss_type = self.model_cfg.get('loss_type', 'ce')
            supervised_args['loss_type'] = loss_type
            if 'loss' not in self.model_cfg:
                try:
                    # Ensure built-in losses are imported and registered.
                    import src.losses  # noqa: F401
                    from src.registry import LOSS_REGISTRY

                    hardcoded_loss_types = {
                        'ce',
                        'bce',
                        'l1',
                        'l2',
                        'bce_l1',
                        'bce_l2',
                        'bce_huber',
                        'bce_topo',
                    }
                    params = self.model_cfg.get('params')
                    if (
                        isinstance(params, dict)
                        and loss_type not in hardcoded_loss_types
                        and loss_type in LOSS_REGISTRY
                    ):
                        supervised_args['loss'] = {'name': loss_type, 'params': params}
                except Exception:
                    pass
            # Optional: loss registry 사용
            if 'loss_name' in self.model_cfg:
                supervised_args['loss_name'] = self.model_cfg['loss_name']
            if 'loss_kwargs' in self.model_cfg:
                supervised_args['loss_kwargs'] = self.model_cfg['loss_kwargs']
            if 'loss' in self.model_cfg:
                supervised_args['loss'] = self.model_cfg['loss']
            
            return SupervisedModel(**supervised_args)
        else:  # diffusion
            return DiffusionModel(
                **common_args,
                dim=self.model_cfg.get('dim', 64),
                timesteps=self.model_cfg.get('timesteps', 1000),
                soft_label_type=self.model_cfg.get('soft_label_type', 'none'),
                soft_label_cache=self.model_cfg.get('soft_label_cache', True),
                soft_label_fg_max=self.model_cfg.get('soft_label_fg_max', 11),
                soft_label_thickness_max=self.model_cfg.get('soft_label_thickness_max', 13),
                soft_label_kernel_ratio=self.model_cfg.get('soft_label_kernel_ratio', 0.1),
                use_ema=self.model_cfg.get('use_ema', True),
                ema_decay=self.model_cfg.get('ema_decay', 0.9999),
                num_ensemble=self.model_cfg.get('num_ensemble', 1),
                log_image_enabled=self.model_cfg.get('log_image_enabled', False),
                log_image_names=self.model_cfg.get('log_image_names', None),
                loss_type=self.model_cfg.get('loss_type', 'hybrid'),
            )

    def _create_callbacks(self, exp_dir: Path):
        """Create callbacks."""
        callbacks = [
            ModelCheckpoint(
                dirpath=exp_dir / "checkpoints",
                filename='best',
                monitor='val/dice',
                mode='max',
                save_last=True,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        if not self.debug:
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=self.model_cfg.get('learning_rate', 1e-4) / 10
                )
            )

        # Always add compact epoch summary logger (rank0 only).
        max_epochs = self.trainer_cfg.get('max_epochs', 300)
        callbacks.append(EpochSummaryLogger(max_epochs=max_epochs))

        return callbacks

    def _create_logger(self, exp_dir: Path, experiment_id: str = None):
        """Create logger with descriptive name."""
        # TensorBoardLogger structure: save_dir/name/version/
        # We want: exp_dir/tensorboard/ (flat structure)
        # So set name to "tensorboard" and version to empty string
        # This creates: exp_dir/tensorboard/
        return EnhancedTensorBoardLogger(
            save_dir=exp_dir,
            name="tensorboard",
            version="",  # Empty version to avoid nested directories
        )

    def _create_trainer(self, callbacks, logger):
        """Create trainer."""
        # Extract precision (handle both "32" and "32-true" formats)
        # Environment variable override for DDP script compatibility
        precision_str = os.environ.get('DDP_PRECISION') or str(self.trainer_cfg.get('precision', '32-true'))

        # GPU/Device configuration
        # - devices: number of GPUs or list of GPU indices (default: 1)
        # - strategy: 'auto', 'ddp', 'ddp_find_unused_parameters_true', etc.
        # Environment variables override config for DDP script compatibility
        devices_env = os.environ.get('DDP_DEVICES')
        if devices_env:
            devices = int(devices_env) if devices_env.lstrip('-').isdigit() else devices_env
        else:
            devices = self.trainer_cfg.get('devices', 1)
        
        strategy = os.environ.get('DDP_STRATEGY') or self.trainer_cfg.get('strategy', 'auto')

        progress_bar = self.trainer_cfg.get('enable_progress_bar', False)
        tqdm_refresh = self.trainer_cfg.get('progress_bar_refresh_rate', 0)

        return L.Trainer(
            max_epochs=self.trainer_cfg.get('max_epochs', 300),
            accelerator=self.trainer_cfg.get('accelerator', 'gpu'),
            devices=devices,
            strategy=strategy,
            precision=precision_str,
            callbacks=callbacks + ([TQDMProgressBar(refresh_rate=tqdm_refresh or 1)] if progress_bar else []),
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=progress_bar,
            log_every_n_steps=self.trainer_cfg.get('log_every_n_steps', 50),
            check_val_every_n_epoch=self.trainer_cfg.get('check_val_every_n_epoch', 5),
            gradient_clip_val=self.trainer_cfg.get('gradient_clip_val', None),
            accumulate_grad_batches=self.trainer_cfg.get('accumulate_grad_batches', 1),
            limit_train_batches=self.trainer_cfg.get('limit_train_batches', 1.0),
            limit_val_batches=self.trainer_cfg.get('limit_val_batches', 1.0),
        )
