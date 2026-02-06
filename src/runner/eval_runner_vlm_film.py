"""
Unified evaluation runner for all models.

This module provides a centralized evaluation system that:
- Automatically finds best checkpoints from experiments/
- Loads models and runs inference on test data
- Computes metrics and saves results
- Supports both supervised and diffusion models
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
import pandas as pd
import torch
import yaml

from src.experiment.tracker import ExperimentTracker
from src.loggers import PredictionLogger
from src.registry.datasets import DATASET_REGISTRY, get_dataset_info
from src.registry.models import MODEL_REGISTRY, get_model_info


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model: str
    dataset: str
    metrics: Dict[str, float]
    checkpoint_path: str
    experiment_id: str


class EvalRunnerVLMFiLM:
    """
    Evaluation runner for VLM-FiLM models.
    
    Example:
        >>> runner = EvalRunnerVLMFiLM(dataset='octa500_3m')
        >>> results = runner.evaluate_all_models()
        >>> runner.save_results(results, 'results/eval_results.csv')
    """

    def __init__(
        self,
        dataset: str,
        output_dir: str = "results/evaluation",
        gpu: Optional[int] = None,
        save_predictions: bool = False,
        save_intermediate: bool = False,
        intermediate_t: Optional[str] = None,
        intermediate_steps: Optional[str] = None,
        intermediate_max_samples: int = 0,
    ):
        """
        Initialize evaluation runner.
        
        Args:
            dataset: Dataset name (e.g., 'octa500_3m')
            output_dir: Directory to save evaluation results
            gpu: GPU index to use (None for CPU)
            save_predictions: Whether to save prediction images
        """
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu = gpu
        self.save_predictions = save_predictions
        self.save_intermediate = bool(save_intermediate)
        self.intermediate_t = self._parse_csv_floats(intermediate_t) if intermediate_t else None
        self.intermediate_steps = self._parse_csv_ints(intermediate_steps) if intermediate_steps else None
        self.intermediate_max_samples = int(intermediate_max_samples)

        # Get dataset info
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset_info = get_dataset_info(dataset)

        # Initialize tracker
        self.tracker = ExperimentTracker()

    @staticmethod
    def _parse_csv_floats(value: str) -> Optional[List[float]]:
        if not value:
            return None
        items = []
        for part in value.split(','):
            part = part.strip()
            if not part:
                continue
            items.append(float(part))
        return items or None

    @staticmethod
    def _parse_csv_ints(value: str) -> Optional[List[int]]:
        if not value:
            return None
        items = []
        for part in value.split(','):
            part = part.strip()
            if not part:
                continue
            items.append(int(part))
        return items or None

    def find_best_checkpoint(self, model: str) -> Optional[Path]:
        """
        Find best checkpoint for a model on this dataset.
        
        Searches in experiments/{model}/{dataset}/ for the best checkpoint.
        
        Args:
            model: Model name
            
        Returns:
            Path to best checkpoint or None if not found
        """
        exp_dir = Path("experiments") / model / self.dataset
        if not exp_dir.exists():
            return None

        # Find all experiment directories
        exp_runs = sorted(exp_dir.glob(f"{model}_{self.dataset}_*"))
        if not exp_runs:
            return None

        # Search for best.ckpt in the latest run first
        for run_dir in reversed(exp_runs):
            best_ckpt = run_dir / "checkpoints" / "best.ckpt"
            if best_ckpt.exists():
                return best_ckpt

        return None

    def get_data_module(
        self,
        data_cfg: Optional[dict] = None,
        label_subdir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """Create data module for current dataset.

        Args:
            data_cfg: Data config dict (from training config.yaml) to mirror train settings.
            label_subdir: Optional label subdirectory override (e.g., 'label_sauna').
            dataset_name: Optional dataset override (uses training config if provided).
        """
        import inspect

        dataset_name = dataset_name or self.dataset
        info = get_dataset_info(dataset_name)
        dm_cls = info.class_ref
        sig = inspect.signature(dm_cls.__init__)

        # Build kwargs from config with dataset defaults as fallback.
        data_cfg = data_cfg or {}
        params = set(sig.parameters)

        kwargs = {
            'train_dir': data_cfg.get('train_dir', info.default_train_dir),
            'val_dir': data_cfg.get('val_dir', info.default_val_dir),
            'test_dir': data_cfg.get('test_dir', info.default_test_dir),
            'crop_size': data_cfg.get('crop_size', info.default_crop_size),
            # Keep training batch size from config for consistency, even though eval uses bs=1.
            'train_bs': data_cfg.get('train_bs', info.default_batch_size),
        }

        if 'num_samples_per_image' in data_cfg:
            kwargs['num_samples_per_image'] = data_cfg['num_samples_per_image']
        if 'label_subdir' in data_cfg:
            kwargs['label_subdir'] = data_cfg['label_subdir']
        elif label_subdir is not None:
            kwargs['label_subdir'] = label_subdir
        if 'use_sauna_transform' in data_cfg:
            kwargs['use_sauna_transform'] = data_cfg['use_sauna_transform']
        if 'train_manual_dir' in data_cfg:
            kwargs['train_manual_dir'] = data_cfg['train_manual_dir']
        if 'train_sam_dir' in data_cfg:
            kwargs['train_sam_dir'] = data_cfg['train_sam_dir']
        if 'use_sam' in data_cfg:
            kwargs['use_sam'] = data_cfg['use_sam']

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        return dm_cls(**filtered_kwargs)

    def _load_experiment_config(self, checkpoint_path: Path) -> Optional[dict]:
        """Load saved training config for the experiment that produced this checkpoint."""
        exp_dir = checkpoint_path.parent.parent
        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}

        exp_id = exp_dir.name
        return self.tracker.get_experiment_config(exp_id)

    def _is_vlm_checkpoint(self, checkpoint_path: Path) -> bool:
        """Check if checkpoint is a VLM model by inspecting hparams."""
        try:
            ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            hparams = ckpt.get('hyper_parameters', {})
            return hparams.get('use_vlm_film', False)
        except Exception as e:
            print(f"⚠️  Could not check if checkpoint is VLM model: {e}")
            # Default to VLM since this is VLM eval runner
            return True
    
    def _get_vlm_film_stages_from_checkpoint(self, checkpoint_path: Path) -> list | None:
        """Detect VLM Film decoder stages from checkpoint.
        
        Priority:
        1. vlm_film_trained_stages metadata (most reliable - actual trained stages)
        2. hparams.vlm_film_decoder_stages (if saved)
        3. Legacy: 2 FiLM heads = stages [0, 1] (old checkpoints with fallback)
        """
        try:
            ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            # Priority 1: Check for trained stages metadata (new checkpoints)
            trained_stages = ckpt.get('vlm_film_trained_stages')
            if trained_stages is not None:
                print(f"   Detected VLM Film stages from checkpoint metadata: {trained_stages}")
                return sorted(trained_stages)
            
            # Priority 2: Try hparams
            hparams = ckpt.get('hyper_parameters', {})
            vlm_stages = hparams.get('vlm_film_decoder_stages')
            if vlm_stages is not None:
                print(f"   Detected VLM Film stages from hparams: {vlm_stages}")
                return vlm_stages
            
            # Priority 3: Legacy checkpoint - infer stages from FiLM head output dimensions
            # Decoder stage channels: Stage0=128, Stage1=96, Stage2=64, Stage3=32
            # FiLM outputs gamma+beta: Stage0=256, Stage1=192, Stage2=128, Stage3=64
            state_dict = ckpt.get('state_dict', {})
            head_output_dims = {}
            for key in state_dict.keys():
                # Get final layer output dimension: vlm_film_heads.{i}.mlp.2.weight shape is [out, in]
                if key.startswith('vlm_film_heads.') and key.endswith('.mlp.2.weight'):
                    head_idx = int(key.split('vlm_film_heads.')[1].split('.')[0])
                    out_dim = state_dict[key].shape[0]
                    head_output_dims[head_idx] = out_dim
            
            if head_output_dims:
                # Map output dimensions to decoder stages
                dim_to_stage = {256: 0, 192: 1, 128: 2, 64: 3}
                stage_mapping = {}
                for head_idx, out_dim in sorted(head_output_dims.items()):
                    stage = dim_to_stage.get(out_dim)
                    if stage is not None:
                        stage_mapping[head_idx] = stage
                
                if stage_mapping:
                    vlm_stages = sorted(stage_mapping.values())
                    print(f"   Legacy checkpoint detected: {len(head_output_dims)} heads")
                    print(f"   Head output dims: {dict(sorted(head_output_dims.items()))}")
                    print(f"   Inferred stages: {vlm_stages}")
                    return vlm_stages
            
            return None
        except Exception as e:
            print(f"⚠️  Could not detect VLM Film stages: {e}")
            return None

    def evaluate_model(self, model_name: str, checkpoint_path: Optional[Path] = None) -> Optional[EvaluationResult]:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model to evaluate
            checkpoint_path: Optional path to specific checkpoint (overrides auto-detection)
            
        Returns:
            EvaluationResult or None if evaluation failed
        """
        # Check if model exists
        if model_name not in MODEL_REGISTRY:
            print(f"❌ Unknown model: {model_name}")
            return None

        model_info = get_model_info(model_name)

        # Find checkpoint (use provided path or auto-detect)
        if checkpoint_path is None:
            checkpoint_path = self.find_best_checkpoint(model_name)
        if not checkpoint_path:
            print(f"❌ No checkpoint found for {model_name} on {self.dataset}")
            return None

        print(f"📊 Evaluating {model_name} on {self.dataset}...")
        print(f"   Checkpoint: {checkpoint_path}")

        try:
            # Load model based on task type (allow full checkpoint load for PyTorch 2.6+)
            # Use strict=False to allow missing/unexpected keys (e.g., LayerNorm changes)
            weights_only = False
            strict = False
            # Always load checkpoints on CPU to avoid invalid device ids baked into checkpoints.
            map_location = torch.device("cpu")
            if model_info.task == 'supervised':
                from src.archs.supervised_model import SupervisedModel
                model = SupervisedModel.load_from_checkpoint(
                    str(checkpoint_path),
                    map_location=map_location,
                    weights_only=weights_only,
                    strict=strict,
                )
            elif model_info.task == 'flow':
                # Auto-detect VLM vs non-VLM checkpoint
                is_vlm = self._is_vlm_checkpoint(checkpoint_path)
                if is_vlm:
                    print(f"   Detected VLM checkpoint, loading FlowModelVLMFiLM...")
                    from src.archs.flow_model_vlm_film import FlowModelVLMFiLM
                    # FiLM 적용: 디코더 stage 0,1,2,3 전부 사용. 체크포인트에서 감지되면 그대로, 없으면 [0,1,2,3].
                    vlm_stages = self._get_vlm_film_stages_from_checkpoint(checkpoint_path)
                    if not vlm_stages:
                        vlm_stages = [0, 1, 2, 3]
                        print(f"   Using default vlm_film_decoder_stages: {vlm_stages}")
                    model = FlowModelVLMFiLM.load_from_checkpoint(
                        str(checkpoint_path),
                        map_location=map_location,
                        weights_only=weights_only,
                        strict=strict,
                        vlm_film_decoder_stages=vlm_stages,
                    )
                    loaded_stages = getattr(model.hparams, "vlm_film_decoder_stages", vlm_stages)
                    model_internal_stages = getattr(model, "_vlm_film_stage_indices", None)
                    print(f"   Loaded with vlm_film_decoder_stages={loaded_stages}")
                    print(f"   Model internal _vlm_film_stage_indices={model_internal_stages}")
                else:
                    print(f"   Detected non-VLM checkpoint, loading FlowModel...")
                    from src.archs.flow_model import FlowModel
                    model = FlowModel.load_from_checkpoint(
                        str(checkpoint_path),
                        map_location=map_location,
                        weights_only=weights_only,
                        strict=strict,
                    )
            else:  # diffusion
                from src.archs.diffusion_model import DiffusionModel
                model = DiffusionModel.load_from_checkpoint(
                    str(checkpoint_path),
                    map_location=map_location,
                    weights_only=weights_only,
                    strict=strict,
                )

            # Load training config (required for consistent evaluation)
            config = self._load_experiment_config(checkpoint_path)
            if config is None:
                print(
                    f"❌ Missing training config for {checkpoint_path}. "
                    "Expected config.yaml in experiment dir or entry in experiments.json."
                )
                return None

            data_cfg = config.get('data', {}) or {}
            label_subdir = data_cfg.get('label_subdir', None)
            config_dataset = data_cfg.get('name', self.dataset)
            if config_dataset != self.dataset:
                print(f"⚠️  Eval dataset override: {self.dataset} -> {config_dataset} (from config)")

            # Setup data (use training config settings when available)
            data_module = self.get_data_module(
                data_cfg=data_cfg,
                label_subdir=label_subdir,
                dataset_name=config_dataset,
            )
            data_module.setup("test")

            # Extract experiment ID from checkpoint path (before creating logger)
            exp_id = checkpoint_path.parent.parent.name

            if self.save_intermediate:
                intermediate_dir = self.output_dir / model_name / exp_id / "intermediate"
                model.eval_save_intermediate = True
                model.eval_intermediate_dir = str(intermediate_dir)
                model.eval_intermediate_t = self.intermediate_t
                model.eval_intermediate_steps = self.intermediate_steps
                model.eval_intermediate_max_samples = self.intermediate_max_samples

            # Create logger if saving predictions
            logger = None
            if self.save_predictions:
                # Include experiment_id in prediction path to avoid overwriting
                pred_dir = self.output_dir / model_name / exp_id / "predictions"
                logger = PredictionLogger(
                    save_dir=str(pred_dir.parent),
                    name="predictions",
                    version=None
                )

            # Create trainer
            trainer = L.Trainer(
                accelerator="gpu" if self.gpu is not None else "cpu",
                devices=[self.gpu] if self.gpu is not None else 1,
                logger=logger,
                enable_checkpointing=False,
                enable_progress_bar=True,
            )

            # Run evaluation
            results = trainer.test(model, data_module)

            if results and len(results) > 0:
                metrics = results[0]

                print(f"✅ {model_name}: Dice={metrics.get('test/dice', 0):.4f}, "
                      f"IoU={metrics.get('test/iou', 0):.4f}")

                return EvaluationResult(
                    model=model_name,
                    dataset=config_dataset,
                    metrics=metrics,
                    checkpoint_path=str(checkpoint_path),
                    experiment_id=exp_id
                )

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

        return None

    def evaluate_models(self, models: List[str]) -> List[EvaluationResult]:
        """
        Evaluate multiple models.
        
        Args:
            models: List of model names
            
        Returns:
            List of evaluation results
        """
        results = []

        print(f"\n{'='*80}")
        print(f"Evaluating {len(models)} models on {self.dataset}")
        print(f"{'='*80}\n")

        for model_name in models:
            result = self.evaluate_model(model_name)
            if result:
                results.append(result)

        return results

    def evaluate_all_models(self) -> List[EvaluationResult]:
        """Evaluate all available models."""
        models = list(MODEL_REGISTRY.keys())
        return self.evaluate_models(models)

    def save_results(
        self,
        results: List[EvaluationResult],
        filename: Optional[str] = None,
        append: bool = False
    ) -> Path:
        """
        Save evaluation results to CSV.
        
        Args:
            results: List of evaluation results
            filename: Output filename (default: evaluation_{dataset}.csv)
            append: If True, append to existing file instead of overwriting
            
        Returns:
            Path to saved CSV file
        """
        if not results:
            print("⚠️  No results to save!")
            return None

        # Convert to DataFrame
        rows = []
        for result in results:
            row = {
                'Model': result.model,
                'Dataset': result.dataset,
                'Experiment_ID': result.experiment_id,
            }
            # Add metrics (remove test/ prefix for cleaner column names)
            for key, value in result.metrics.items():
                clean_key = key.replace('test/', '')
                row[clean_key] = value
            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns: Model, Dataset, then metrics
        metric_cols = [col for col in df.columns
                      if col not in ['Model', 'Dataset', 'Experiment_ID']]
        df = df[['Model', 'Dataset'] + metric_cols + ['Experiment_ID']]

        # Determine filename
        if filename is None:
            # If single result, include experiment_id in filename
            if len(results) == 1:
                exp_id = results[0].experiment_id
                model_name = results[0].model
                filename = f"evaluation_{model_name}_{exp_id}_{self.dataset}.csv"
            else:
                filename = f"evaluation_{self.dataset}.csv"

        output_path = self.output_dir / filename
        
        # Append or overwrite
        if append and output_path.exists():
            existing_df = pd.read_csv(output_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"📝 Appended results to existing file: {output_path}")
        else:
            df.to_csv(output_path, index=False, float_format='%.6f')

        # Print results
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}")
        print(f"✅ Results saved to: {output_path}\n")

        return output_path
