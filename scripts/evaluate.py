"""
Unified evaluation script for all models.

Usage:
    # Evaluate all models on octa500_3m
    uv run python scripts/evaluate.py --data octa500_3m
    
    # Evaluate specific models
    uv run python scripts/evaluate.py --data octa500_3m --models csnet,dscnet,medsegdiff
    
    # Save predictions
    uv run python scripts/evaluate.py --data octa500_3m --save-predictions
    
    # Use specific GPU
    uv run python scripts/evaluate.py --data octa500_6m --gpu 0
    
    # Custom output directory
    uv run python scripts/evaluate.py --data rossa --output results/my_eval
    
    # Evaluate with specific checkpoint (for soft label experiments)
    uv run python scripts/evaluate.py --data xca --models csnet --checkpoint experiments/csnet_label_smooth/xca/logs/checkpoints/best.ckpt
"""

import autorootcwd
import click
from pathlib import Path

from src.runner.eval_runner import EvalRunner
from src.registry.models import MODEL_REGISTRY
from src.registry.datasets import DATASET_REGISTRY


@click.command()
@click.option('--data', required=True, help='Dataset name (e.g., octa500_3m, octa500_6m, rossa, xca)')
@click.option('--models', default=None, help='Comma-separated model names (default: all models)')
@click.option('--output', default='results/evaluation', help='Output directory for results')
@click.option('--gpu', default=None, type=int, help='GPU index to use (default: None=CPU)')
@click.option('--save-predictions/--no-save-predictions', default=True, help='Save prediction images (default: True)')
@click.option('--checkpoint', default=None, help='Path to specific checkpoint file (overrides auto-detection)')
def main(data, models, output, gpu, save_predictions, checkpoint):
    """Evaluate trained models on test data."""
    
    # Validate dataset
    if data not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        click.echo(f"❌ Unknown dataset: {data}")
        click.echo(f"   Available datasets: {available}")
        return
    
    # Parse model list
    if models:
        model_list = [m.strip() for m in models.split(',')]
        # Validate models
        invalid = [m for m in model_list if m not in MODEL_REGISTRY]
        if invalid:
            available = ', '.join(MODEL_REGISTRY.keys())
            click.echo(f"❌ Unknown models: {', '.join(invalid)}")
            click.echo(f"   Available models: {available}")
            return
    else:
        model_list = None  # Will evaluate all models
    
    # Validate checkpoint if provided
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            click.echo(f"❌ Checkpoint not found: {checkpoint}")
            return
        if not model_list or len(model_list) != 1:
            click.echo("❌ When using --checkpoint, you must specify exactly one model with --models")
            return
    
    # Create runner
    runner = EvalRunner(
        dataset=data,
        output_dir=output,
        gpu=gpu,
        save_predictions=save_predictions
    )
    
    # Run evaluation
    if checkpoint:
        # Single model with specific checkpoint
        results = runner.evaluate_model(model_list[0], checkpoint_path=Path(checkpoint))
        results = [results] if results else []
    elif model_list:
        results = runner.evaluate_models(model_list)
    else:
        results = runner.evaluate_all_models()
    
    # Save results
    if results:
        runner.save_results(results)
    else:
        click.echo("\n⚠️  No models were successfully evaluated!")


if __name__ == "__main__":
    main()
