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


class EvalRunner:
    """
    Unified evaluation runner for all models.
    
    Example:
        >>> runner = EvalRunner(dataset='octa500_3m')
        >>> results = runner.evaluate_all_models()
        >>> runner.save_results(results, 'results/eval_results.csv')
    """

    def __init__(
        self,
        dataset: str,
        output_dir: str = "results/evaluation",
        gpu: Optional[int] = None,
        save_predictions: bool = False,
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

        # Get dataset info
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset_info = get_dataset_info(dataset)

        # Initialize tracker
        self.tracker = ExperimentTracker()

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

    def get_data_module(self, label_subdir: Optional[str] = None):
        """Create data module for current dataset.
        
        Args:
            label_subdir: Label subdirectory (e.g., 'label_sauna'). 
                         If None, uses default 'label'.
        """
        import inspect

        info = self.dataset_info
        dm_cls = info.class_ref
        sig = inspect.signature(dm_cls.__init__)

        # Build kwargs from metadata, respecting the datamodule signature
        kwargs = {}
        params = sig.parameters

        if 'train_dir' in params:
            kwargs['train_dir'] = info.default_train_dir
        if 'val_dir' in params:
            kwargs['val_dir'] = info.default_val_dir
        if 'test_dir' in params:
            kwargs['test_dir'] = info.default_test_dir
        if 'train_manual_dir' in params:
            kwargs['train_manual_dir'] = info.default_train_dir
        if 'train_sam_dir' in params and info.default_train_dir:
            kwargs['train_sam_dir'] = info.default_train_dir.replace('train_manual', 'train_sam')

        if 'crop_size' in params:
            kwargs['crop_size'] = info.default_crop_size
        if 'train_bs' in params:
            kwargs['train_bs'] = 1  # evaluation은 항상 batch_size=1로 고정
        if label_subdir and 'label_subdir' in params:
            kwargs['label_subdir'] = label_subdir

        # Fallback: respect provided defaults in signature if metadata missing
        return dm_cls(**kwargs)

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
            weights_only = False
            if model_info.task == 'supervised':
                from src.archs.supervised_model import SupervisedModel
                model = SupervisedModel.load_from_checkpoint(str(checkpoint_path), weights_only=weights_only, strict=False)
            elif model_info.task == 'flow':
                from src.archs.flow_model import FlowModel
                model = FlowModel.load_from_checkpoint(str(checkpoint_path), weights_only=weights_only, strict=False)
            else:  # diffusion
                from src.archs.diffusion_model import DiffusionModel
                model = DiffusionModel.load_from_checkpoint(str(checkpoint_path), weights_only=weights_only, strict=False)

            # Extract label_subdir from checkpoint config if available
            label_subdir = None
            config_path = checkpoint_path.parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    label_subdir = config.get('data', {}).get('label_subdir', None)

            # Setup data (use label_subdir from config if available)
            data_module = self.get_data_module(label_subdir=label_subdir)
            data_module.setup("test")

            # Extract experiment ID from checkpoint path (before creating logger)
            exp_id = checkpoint_path.parent.parent.name

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
                    dataset=self.dataset,
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
