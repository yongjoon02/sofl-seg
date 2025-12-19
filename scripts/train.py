#!/usr/bin/env python
"""
Config-based training script.

Supports both YAML config files and CLI arguments.
CLI arguments override config values.

Examples:
    # Use config file only
    uv run python scripts/train.py --config configs/octa500_3m_supervised_models.yaml
    
    # Override with CLI arguments
    uv run python scripts/train.py --config configs/octa500_3m_supervised_models.yaml \\
        --model dscnet --batch-size 64 --lr 0.001
    
    # CLI only (uses default config)
    uv run python scripts/train.py --model csnet --data octa500_3m
    
    # Debug mode
    uv run python scripts/train.py --config configs/octa500_3m_supervised_models.yaml --debug
"""

import os
os.environ['NCCL_P2P_DISABLE'] = '1'

import torch
torch.set_float32_matmul_precision('medium')

import argparse
import autorootcwd

from src.runner import TrainRunner
from src.registry import list_models, list_datasets
from src.utils.config import load_config, merge_config_with_args, get_default_config_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train vessel segmentation models with YAML configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file (primary method)
    parser.add_argument('--config', type=str,
                       help='Path to YAML config file')
    parser.add_argument('--lr-find', action='store_true',
                       help='Run Lightning LR finder instead of full training')
    
    # Model/Data (required if no config)
    parser.add_argument('--model', type=str,
                       help=f'Model name (overrides config). Available: {", ".join(list_models())}')
    parser.add_argument('--data', type=str,
                       help=f'Dataset name (overrides config). Available: {", ".join(list_datasets())}')
    
    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int,
                       help='Max epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--crop-size', type=int,
                       help='Crop size')
    parser.add_argument('--gpu', type=int,
                       help='GPU device ID')
    
    # Diffusion-specific (override config)
    parser.add_argument('--soft-label', type=str,
                       choices=['none', 'thickness', 'faz'],
                       help='Soft label type for diffusion models')
    parser.add_argument('--soft-label-fg-max', type=int,
                       help='Max foreground distance for soft labels')
    parser.add_argument('--soft-label-thickness-max', type=int,
                       help='Max thickness value for soft labels')
    parser.add_argument('--timesteps', type=int,
                       help='Diffusion timesteps')
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable EMA for diffusion models')
    parser.add_argument('--ema-decay', type=float,
                       help='EMA decay rate')
    parser.add_argument('--ensemble', type=int,
                       help='Number of ensemble samples')
    
    # Experiment control
    parser.add_argument('--tag', type=str,
                       help='Experiment tag')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (2 epochs, limited batches)')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--log-image', action='store_true',
                       help='Enable image logging to TensorBoard')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    if args.config:
        # Load from specified file
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    elif args.model and args.data:
        # Try to find default config
        default_config_path = get_default_config_path(args.model, args.data)
        if default_config_path and default_config_path.exists():
            print(f"Using default config: {default_config_path}")
            config = load_config(str(default_config_path))
        else:
            # Build minimal config from CLI
            print("No config file found, using CLI arguments only")
            config = {
                'model': {'arch_name': args.model},
                'data': {'name': args.data},
                'trainer': {}
            }
    else:
        parser.error("Either --config or both --model and --data are required")

    # Merge CLI overrides
    config = merge_config_with_args(config, args)
    if args.lr_find:
        # Flag for TrainRunner to trigger lr finder mode
        config.setdefault('trainer', {})
        config['trainer']['lr_find'] = True
    
    # Print final config
    print("\n" + "="*60)
    print("Final Configuration:")
    print("="*60)
    print(f"Model: {config['model']['arch_name']}")
    print(f"Dataset: {config['data']['name']}")
    print(f"Batch size: {config['data'].get('train_bs', 'N/A')}")
    print(f"Learning rate: {config['model'].get('learning_rate', 'N/A')}")
    print(f"Max epochs: {config['trainer'].get('max_epochs', 'N/A')}")
    if config.get('debug'):
        print("⚠️  DEBUG MODE ENABLED")
    print("="*60 + "\n")
    
    # Create runner and train
    runner = TrainRunner(config)
    runner.run()


if __name__ == '__main__':
    main()
