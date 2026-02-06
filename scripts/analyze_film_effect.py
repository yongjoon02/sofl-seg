"""
Analyze VLM-FiLM Effect on Decoder Features

Purpose:
    Measure the impact of VLM-FiLM conditioning by comparing decoder features
    BEFORE and AFTER FiLM modulation using PyTorch forward hooks.

Why hooks?
    - Non-invasive: No modification to training/evaluation code
    - Precise: Captures exact intermediate tensors during forward pass
    - Clean: Hooks can be registered and removed externally

Why eval() + no_grad()?
    - Removes training-specific behavior (dropout, batch norm updates)
    - Disables gradient computation (faster, less memory)
    - Reflects actual inference behavior

What we extract:
    - x_pre: Decoder stage feature AFTER junction gate, BEFORE FiLM
    - x_post: Decoder stage feature AFTER FiLM application
    - Both from the SAME forward pass

Analysis metrics:
    - Channel-wise activation energy: A = sqrt(sum_{spatial} x^2)
    - Difference map: ΔA = A_post - A_pre
    - Statistics: mean, std, max change across channels
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
import autorootcwd  # noqa: F401

from src.archs.flow_model import FlowModel
from src.data import XCA_DataModule, OCTA500_3M_DataModule, OCTA500_6M_DataModule, ROSSA_DataModule

# Import losses to populate LOSS_REGISTRY before loading checkpoint
from src.losses import FlowMatchingLoss  # noqa: F401


class FiLMFeatureExtractor:
    """
    Extract features before and after FiLM using forward hooks.
    
    Hook strategy:
        - Pre-hook on apply_vlm_film: captures x_pre
        - Post-hook on apply_vlm_film: captures x_post
        - Both hooks fire in the same forward pass
    
    Why this works:
        - apply_vlm_film is the exact function that applies FiLM
        - Pre-hook captures input (x before FiLM)
        - Post-hook captures output (x after FiLM)
        - Junction gate is already applied in x_pre (part of the input)
    """
    
    def __init__(self):
        self.x_pre = None
        self.x_post = None
        self.gamma = None
        self.beta = None
        self.junction_gate = None
        self.stage_idx = None
        self.hooks = []
        # For cumulative effect analysis
        self.x_first_pre = None
        self.x_last_post = None
        self.first_stage = None
        self.last_stage = None
    
    def _find_apply_vlm_film_calls(self, module: nn.Module):
        """
        Find where apply_vlm_film is called in the decoder.
        
        Strategy: Patch the apply_vlm_film function temporarily to capture args.
        This is cleaner than trying to hook arbitrary decoder blocks.
        """
        # We'll use a wrapper approach: monkey-patch apply_vlm_film
        pass
    
    def register_hooks(self, model: nn.Module, target_stage: int = 1, first_stage: int = None, last_stage: int = None):
        """
        Register hooks to capture FiLM input/output at target decoder stage.
        
        Args:
            model: FlowModel (LightningModule)
            target_stage: Decoder stage index to monitor (0=stage0, 1=stage1, ...) - single stage analysis
            first_stage: First stage to capture (for cumulative effect) - captures x_pre at this stage
            last_stage: Last stage to capture (for cumulative effect) - captures x_post at this stage
        
        Implementation note:
            Since apply_vlm_film is a function (not a module), we can't directly
            hook it. Instead, we hook the decoder blocks and detect when FiLM
            is applied by checking vlm_cond presence.
            
            Better approach: Patch apply_vlm_film globally during analysis.
        """
        from src.archs.components import vlm_film
        
        # Store original function
        self.original_apply_vlm_film = vlm_film.apply_vlm_film
        self.target_stage = target_stage
        self.first_stage = first_stage
        self.last_stage = last_stage
        
        def patched_apply_vlm_film(
            x, vlm_cond, vlm_film_heads, stage_idx, module,
            apply_point="legacy_post_block2",
            junction_gate=None,
            junction_warmup_active=False
        ):
            """Wrapper that captures x_pre, x_post, and calls original."""
            # Debug: print stage_idx to verify it's being called
            # print(f"[DEBUG] apply_vlm_film called with stage_idx={stage_idx}")
            
            # Capture x_pre (input to FiLM)
            if stage_idx == self.target_stage:
                self.x_pre = x.detach().clone()
                self.stage_idx = stage_idx
                self.junction_gate = junction_gate.detach().clone() if junction_gate is not None else None
                
                # Extract gamma, beta if possible
                if vlm_cond is not None and vlm_film_heads is not None:
                    cond_vec = vlm_cond['cond_vec']
                    gamma, beta = vlm_film_heads[stage_idx](cond_vec)
                    self.gamma = gamma.detach().clone()
                    self.beta = beta.detach().clone()
            
            # Cumulative effect: capture first stage pre
            if self.first_stage is not None and stage_idx == self.first_stage:
                self.x_first_pre = x.detach().clone()
                print(f"[DEBUG] Captured x_first_pre at stage {stage_idx}, shape={x.shape}")
            
            # Call original function
            x_out, payload = self.original_apply_vlm_film(
                x, vlm_cond, vlm_film_heads, stage_idx, module,
                apply_point, junction_gate, junction_warmup_active
            )
            
            # Capture x_post (output from FiLM)
            if stage_idx == self.target_stage:
                self.x_post = x_out.detach().clone()
            
            # Cumulative effect: capture last stage post
            if self.last_stage is not None and stage_idx == self.last_stage:
                self.x_last_post = x_out.detach().clone()
                print(f"[DEBUG] Captured x_last_post at stage {stage_idx}, shape={x_out.shape}")
            
            return x_out, payload
        
        # Monkey-patch globally
        vlm_film.apply_vlm_film = patched_apply_vlm_film
        if first_stage is not None and last_stage is not None:
            print(f"[Hook] Registered FiLM monitor for cumulative effect: stage {first_stage} pre → stage {last_stage} post")
        else:
            print(f"[Hook] Registered FiLM monitor at decoder stage {target_stage}")
    
    def remove_hooks(self):
        """Restore original apply_vlm_film function."""
        from src.archs.components import vlm_film
        if hasattr(self, 'original_apply_vlm_film'):
            vlm_film.apply_vlm_film = self.original_apply_vlm_film
            print("[Hook] Removed FiLM monitor")
    
    def compute_activation_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel-wise activation energy.
        
        Args:
            x: Feature tensor (B, C, H, W)
        
        Returns:
            A: Activation energy (B, C) - L2 norm over spatial dims
        """
        # Sum of squares over spatial dimensions
        energy = (x ** 2).sum(dim=(2, 3))  # (B, C)
        # Square root to get L2 norm
        return torch.sqrt(energy + 1e-8)
    
    def analyze(self, cumulative=False):
        """
        Analyze captured features and compute FiLM effect metrics.
        
        Args:
            cumulative: If True, analyze x_first_pre vs x_last_post (cumulative effect)
                       If False, analyze x_pre vs x_post (single stage effect)
        
        Returns:
            dict with analysis results
        """
        if cumulative:
            if self.x_first_pre is None or self.x_last_post is None:
                raise RuntimeError("Cumulative features not captured. Check first_stage and last_stage.")
            x_pre = self.x_first_pre
            x_post = self.x_last_post
            stage_info = f"{self.first_stage} → {self.last_stage}"
        else:
            if self.x_pre is None or self.x_post is None:
                raise RuntimeError("Features not captured. Did you run a forward pass?")
            x_pre = self.x_pre
            x_post = self.x_post
            stage_info = self.stage_idx
        
        results = {}
        
        # Basic statistics
        results['x_pre_shape'] = tuple(x_pre.shape)
        results['x_post_shape'] = tuple(x_post.shape)
        results['stage_idx'] = stage_info
        results['cumulative'] = cumulative
        
        # Activation energy per channel
        A_pre = self.compute_activation_energy(x_pre)   # (B, C)
        A_post = self.compute_activation_energy(x_post) # (B, C)
        
        # Change in activation energy
        delta_A = A_post - A_pre  # (B, C)
        
        results['A_pre'] = A_pre.cpu().numpy()
        results['A_post'] = A_post.cpu().numpy()
        results['delta_A'] = delta_A.cpu().numpy()
        
        # Summary statistics (averaged over batch)
        delta_A_mean = delta_A.mean(dim=0)  # (C,)
        results['delta_A_per_channel'] = delta_A_mean.cpu().numpy()
        results['delta_A_mean'] = delta_A_mean.mean().item()
        results['delta_A_std'] = delta_A.std().item()
        results['delta_A_max'] = delta_A.max().item()
        results['delta_A_min'] = delta_A.min().item()
        
        # Spatial difference map (averaged over channels and batch)
        spatial_diff = (x_post - x_pre).abs()  # (B, C, H, W)
        results['spatial_diff_map'] = spatial_diff.mean(dim=(0, 1)).cpu().numpy()  # (H, W)
        
        # FiLM parameters
        if self.gamma is not None:
            results['gamma_mean'] = self.gamma.mean().item()
            results['gamma_std'] = self.gamma.std().item()
            results['gamma_min'] = self.gamma.min().item()
            results['gamma_max'] = self.gamma.max().item()
        
        if self.beta is not None:
            results['beta_mean'] = self.beta.mean().item()
            results['beta_std'] = self.beta.std().item()
            results['beta_min'] = self.beta.min().item()
            results['beta_max'] = self.beta.max().item()
        
        # Junction gate statistics
        if self.junction_gate is not None:
            gate = self.junction_gate
            results['gate_mean'] = gate.mean().item()
            results['gate_std'] = gate.std().item()
            results['gate_min'] = gate.min().item()
            results['gate_max'] = gate.max().item()
            results['gate_zero_ratio'] = (gate == 0.0).float().mean().item()
            
            # Compute FiLM effect separately on junction vs non-junction regions
            # Resize gate to match feature map resolution if needed
            if gate.shape[2:] != self.x_pre.shape[2:]:
                gate_resized = torch.nn.functional.interpolate(
                    gate, size=self.x_pre.shape[2:], mode='nearest'
                )
            else:
                gate_resized = gate
            
            junction_mask = (gate_resized < 0.5)  # gate=0 at junctions
            non_junction_mask = ~junction_mask
            
            # Compute delta only where FiLM is applied (non-junction)
            if non_junction_mask.any():
                delta_at_non_junction = spatial_diff * non_junction_mask.float()
                results['delta_non_junction_mean'] = delta_at_non_junction.sum().item() / non_junction_mask.sum().item()
            
            # Compute delta at junctions (should be ~0 due to gating)
            if junction_mask.any():
                delta_at_junction = spatial_diff * junction_mask.float()
                results['delta_junction_mean'] = delta_at_junction.sum().item() / junction_mask.sum().item()
        
        return results


def load_model_and_data(checkpoint_path: str, config_path: str, device: str = 'cuda'):
    """
    Load trained model and data module.
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to config yaml
        device: Device to load model on
    
    Returns:
        model: FlowModel in eval mode
        datamodule: Data module with test dataloader
    """
    # Load model from checkpoint
    # PyTorch 2.6+ changed weights_only default to True, but we trust our own checkpoints
    model = FlowModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,  # Allow missing keys for flexibility
        weights_only=False,  # Allow full checkpoint loading (optimizer, schedulers, etc.)
    )
    model.eval()
    model.to(device)
    
    print(f"[Model] Loaded from {checkpoint_path}")
    print(f"[Model] VLM-FiLM enabled: {model.use_vlm_film}")
    print(f"[Model] Junction gating enabled: {model.junction_gating_config.get('enabled', False) if hasattr(model, 'junction_gating_config') else False}")
    
    # Load data module
    # Note: We need to parse config to get data settings
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Select appropriate data module based on config
    data_name = config['data'].get('name', 'xca').lower()
    if data_name == 'xca':
        datamodule = XCA_DataModule(
            train_dir=config['data'].get('train_dir'),
            val_dir=config['data'].get('val_dir'),
            test_dir=config['data'].get('test_dir'),
            train_bs=1,  # Single sample for analysis
            crop_size=config['data'].get('image_size', 320),
            num_samples_per_image=1,
            label_subdir=config['data'].get('label_subdir', 'label'),
            use_sauna_transform=config['data'].get('use_sauna_transform', False),
        )
    elif data_name == 'octa500_3m':
        datamodule = OCTA500_3M_DataModule(
            train_dir=config['data'].get('train_dir'),
            val_dir=config['data'].get('val_dir'),
            test_dir=config['data'].get('test_dir'),
            train_bs=1,
            crop_size=config['data'].get('image_size', 304),
            num_samples_per_image=1,
        )
    elif data_name == 'octa500_6m':
        datamodule = OCTA500_6M_DataModule(
            train_dir=config['data'].get('train_dir'),
            val_dir=config['data'].get('val_dir'),
            test_dir=config['data'].get('test_dir'),
            train_bs=1,
            crop_size=config['data'].get('image_size', 304),
            num_samples_per_image=1,
        )
    elif data_name == 'rossa':
        datamodule = ROSSA_DataModule(
            train_dir=config['data'].get('train_dir'),
            val_dir=config['data'].get('val_dir'),
            test_dir=config['data'].get('test_dir'),
            train_bs=1,
            crop_size=config['data'].get('image_size', 512),
            num_samples_per_image=1,
        )
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    
    datamodule.setup('test')
    
    print(f"[Data] Test samples: {len(datamodule.test_dataset)}")
    
    return model, datamodule


def visualize_results(results: dict, save_path: Path):
    """
    Create visualization of FiLM effect.
    
    Args:
        results: Analysis results from FiLMFeatureExtractor.analyze()
        save_path: Directory to save figures
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Channel-wise activation energy change
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Delta A per channel
    ax = axes[0, 0]
    delta_A_per_channel = results['delta_A_per_channel']
    channels = np.arange(len(delta_A_per_channel))
    ax.bar(channels, delta_A_per_channel, color='steelblue', alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('ΔA (Energy Change)')
    ax.set_title('Per-Channel Activation Energy Change')
    ax.grid(True, alpha=0.3)
    
    # (b) Histogram of delta A
    ax = axes[0, 1]
    delta_A_flat = results['delta_A'].flatten()
    ax.hist(delta_A_flat, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('ΔA')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Energy Change\nMean: {results["delta_A_mean"]:.4f}')
    ax.grid(True, alpha=0.3)
    
    # (c) Spatial difference map
    ax = axes[1, 0]
    spatial_diff = results['spatial_diff_map']
    im = ax.imshow(spatial_diff, cmap='hot', interpolation='nearest')
    ax.set_title('Spatial Difference Map\n(averaged over batch & channels)')
    plt.colorbar(im, ax=ax, label='|x_post - x_pre|')
    ax.axis('off')
    
    # (d) FiLM parameters
    ax = axes[1, 1]
    if 'gamma_mean' in results and 'beta_mean' in results:
        labels = ['γ (mean)', 'γ (std)', 'β (mean)', 'β (std)']
        values = [
            results['gamma_mean'],
            results['gamma_std'],
            results['beta_mean'],
            results['beta_std']
        ]
        bars = ax.bar(labels, values, color=['steelblue', 'lightblue', 'coral', 'lightsalmon'])
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Value')
        ax.set_title('FiLM Parameters (γ, β)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top')
    else:
        ax.text(0.5, 0.5, 'FiLM params not captured', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(save_path / 'film_effect_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[Viz] Saved to {save_path / 'film_effect_analysis.png'}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("FiLM EFFECT ANALYSIS SUMMARY")
    print("="*70)
    print(f"Decoder Stage: {results['stage_idx']}")
    print(f"Feature Shape: {results['x_pre_shape']}")
    print(f"\nActivation Energy Change (ΔA):")
    print(f"  Mean:  {results['delta_A_mean']:>10.6f}")
    print(f"  Std:   {results['delta_A_std']:>10.6f}")
    print(f"  Max:   {results['delta_A_max']:>10.6f}")
    print(f"  Min:   {results['delta_A_min']:>10.6f}")
    
    if 'gamma_mean' in results:
        print(f"\nFiLM Parameters:")
        print(f"  γ (gamma): {results['gamma_mean']:.6f} ± {results['gamma_std']:.6f}")
        print(f"             range [{results['gamma_min']:.6f}, {results['gamma_max']:.6f}]")
        print(f"  β (beta):  {results['beta_mean']:.6f} ± {results['beta_std']:.6f}")
        print(f"             range [{results['beta_min']:.6f}, {results['beta_max']:.6f}]")
    
    if 'gate_mean' in results:
        print(f"\nJunction Gate Statistics:")
        print(f"  Mean: {results['gate_mean']:.6f}")
        print(f"  Zero ratio: {results['gate_zero_ratio']*100:.2f}% (junction pixels)")
        print(f"  ΔA at non-junctions: {results.get('delta_non_junction_mean', 0):.6f}")
        print(f"  ΔA at junctions:     {results.get('delta_junction_mean', 0):.6f}")
        print(f"  → Junction gating suppression ratio: {1 - results.get('delta_junction_mean', 0) / (results.get('delta_non_junction_mean', 1e-8) + 1e-8):.2%}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze VLM-FiLM effect on decoder features')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (.yaml)')
    parser.add_argument('--stage', type=int, default=None,
                       help='Decoder stage to analyze (0=stage0, 1=stage1, ...) for single stage analysis')
    parser.add_argument('--first-stage', type=int, default=None,
                       help='First decoder stage for cumulative analysis (captures x_pre at this stage)')
    parser.add_argument('--last-stage', type=int, default=None,
                       help='Last decoder stage for cumulative analysis (captures x_post at this stage)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--output-dir', type=str, default='results/film_analysis',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of test samples to analyze')
    
    args = parser.parse_args()
    
    # Validate arguments
    cumulative_mode = args.first_stage is not None and args.last_stage is not None
    single_mode = args.stage is not None
    
    if not cumulative_mode and not single_mode:
        parser.error("Must specify either --stage (single stage) or both --first-stage and --last-stage (cumulative)")
    if cumulative_mode and single_mode:
        parser.error("Cannot specify both --stage and --first-stage/--last-stage simultaneously")
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("VLM-FiLM EFFECT ANALYZER")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    if cumulative_mode:
        print(f"Mode:       Cumulative (stage {args.first_stage} pre → stage {args.last_stage} post)")
    else:
        print(f"Mode:       Single stage ({args.stage})")
    print(f"Device:     {device}")
    print(f"Output:     {output_dir}")
    print("="*70 + "\n")
    
    # Load model and data
    model, datamodule = load_model_and_data(args.checkpoint, args.config, device)
    test_loader = datamodule.test_dataloader()
    
    # Setup feature extractor
    extractor = FiLMFeatureExtractor()
    if cumulative_mode:
        extractor.register_hooks(model, target_stage=args.last_stage, 
                                first_stage=args.first_stage, last_stage=args.last_stage)
    else:
        extractor.register_hooks(model, target_stage=args.stage)
    
    try:
        # Run forward pass on test samples
        print(f"[Analysis] Running inference on {args.num_samples} sample(s)...")
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= args.num_samples:
                    break
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                images = batch['image']
                
                # Run inference through the model's internal method
                # This triggers the UNet forward pass with VLM-FiLM and hooks
                if hasattr(model, '_infer_sliding') and model.hparams.use_sliding_infer:
                    # Use sliding window inference (more accurate)
                    _ = model._infer_sliding(images)
                elif hasattr(model, '_infer_full'):
                    # Use full image inference
                    _ = model._infer_full(images)
                else:
                    # Fallback: use validation_step logic
                    _ = model.validation_step(batch, i)
                
                print(f"[Analysis] Processed sample {i+1}/{args.num_samples}")
                
                # Analyze (only need one sample for now)
                if i == 0:
                    results = extractor.analyze(cumulative=cumulative_mode)
                    visualize_results(results, output_dir)
        
        print("\n[Success] Analysis complete!")
        
    finally:
        # Always clean up hooks
        extractor.remove_hooks()
        print("[Cleanup] Hooks removed")


if __name__ == '__main__':
    main()
