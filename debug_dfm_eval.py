#!/usr/bin/env python3
"""
Debug script for DFM binary segmentation fragmentation issue.
Instruments the inference pipeline to dump intermediate tensors.

Usage:
    python debug_dfm_eval.py --checkpoint <path> --gpu 0
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Monkey-patch the sampler to dump intermediates
original_sampler_euler = None
original_sampler_heun = None

def instrumented_sampler_euler(model_fn, cond, steps, **kwargs):
    """Instrumented version of sampler_dfm_euler that dumps intermediates."""
    from src.archs.dfm_binary import sampler_dfm_euler as original
    
    # Import here to avoid circular dependency
    eps = kwargs.get('eps', 1e-6)
    
    if cond.dim() == 3:
        cond = cond.unsqueeze(1)
    
    b, _, h, w = cond.shape
    p = torch.full((b, 1, h, w), 0.5, device=cond.device, dtype=cond.dtype)
    x = torch.bernoulli(p)
    
    dt = 1.0 / steps
    
    # Run the loop
    for k in range(steps):
        t = torch.full((b,), k / steps, device=cond.device, dtype=cond.dtype)
        logits = model_fn(x, t, cond)
        p1 = torch.sigmoid(logits)
        p_t = p
        t_view = t.view(-1, 1, 1, 1)
        v = (p1 - p_t) / (1.0 - t_view + eps)
        p_next = torch.clamp(p_t + v * dt, 0.0, 1.0)
        x = torch.bernoulli(p_next)
        p = p_next
    
    # Final prediction at t=1.0 (INSTRUMENTED)
    t_final = torch.ones((b,), device=cond.device, dtype=cond.dtype)
    logits_final = model_fn(x, t_final, cond)
    p_final = torch.sigmoid(logits_final)
    x_final = torch.bernoulli(p_final)
    
    # === DUMP DIAGNOSTICS ===
    debug_dir = Path("debug_dumps")
    debug_dir.mkdir(exist_ok=True)
    
    # Save tensors (first sample only to keep small)
    torch.save(logits_final[0].cpu(), debug_dir / "debug_logits_final.pt")
    torch.save(p_final[0].cpu(), debug_dir / "debug_p_final.pt")
    torch.save(x_final[0].cpu(), debug_dir / "debug_x_final.pt")
    
    # Print stats
    print("\n" + "="*80)
    print("DFM SAMPLER DEBUG OUTPUT (Euler)")
    print("="*80)
    
    def print_tensor_stats(name, tensor):
        print(f"\n{name}:")
        print(f"  shape: {tensor.shape}, dtype: {tensor.dtype}")
        print(f"  min: {tensor.min().item():.6f}, max: {tensor.max().item():.6f}, mean: {tensor.mean().item():.6f}")
        if tensor.dtype in [torch.float32, torch.float16, torch.float64]:
            print(f"  std: {tensor.std().item():.6f}")
            # Percentiles for probability maps
            if 0 <= tensor.min() <= 1 and 0 <= tensor.max() <= 1:
                vals = tensor.flatten().cpu().numpy()
                p1, p50, p99 = np.percentile(vals, [1, 50, 99])
                print(f"  percentiles: p1={p1:.4f}, p50={p50:.4f}, p99={p99:.4f}")
        # Fraction of ones for binary
        if torch.all((tensor == 0) | (tensor == 1)):
            frac_ones = (tensor == 1).float().mean().item()
            print(f"  fraction_ones: {frac_ones:.4f}")
    
    print_tensor_stats("logits_final", logits_final)
    print_tensor_stats("p_final (after sigmoid)", p_final)
    print_tensor_stats("x_final (after bernoulli)", x_final)
    
    print("\n" + "="*80)
    
    return x_final.float()


def instrumented_test_step_dfm(original_fn):
    """Wrapper for _test_step_dfm to dump output_geometry and preds."""
    def wrapper(self, batch, batch_idx):
        # Call original
        result = original_fn(self, batch, batch_idx)
        
        # Intercept outputs if we can
        # Note: This is tricky because test_step doesn't return preds
        # We'll need to hook into the forward pass
        
        return result
    return wrapper


def patch_sampler():
    """Monkey-patch the DFM sampler."""
    import src.archs.dfm_binary as dfm_module
    global original_sampler_euler, original_sampler_heun
    
    original_sampler_euler = dfm_module.sampler_dfm_euler
    original_sampler_heun = dfm_module.sampler_dfm_heun
    
    dfm_module.sampler_dfm_euler = instrumented_sampler_euler
    # For now, only patch Euler; Heun similar


def unpatch_sampler():
    """Restore original sampler."""
    import src.archs.dfm_binary as dfm_module
    if original_sampler_euler:
        dfm_module.sampler_dfm_euler = original_sampler_euler
    if original_sampler_heun:
        dfm_module.sampler_dfm_heun = original_sampler_heun


# Additional instrumentation for _sliding_window_predict
original_sliding_window_predict = None

def instrumented_sliding_window_predict(original_fn):
    """Wrapper to dump output_geometry."""
    def wrapper(self, images):
        output = original_fn(self, images)
        
        # Dump
        debug_dir = Path("debug_dumps")
        debug_dir.mkdir(exist_ok=True)
        torch.save(output[0].cpu(), debug_dir / "debug_output_geometry.pt")
        
        print("\n" + "="*80)
        print("SLIDING WINDOW OUTPUT")
        print("="*80)
        print(f"output_geometry shape: {output.shape}, dtype: {output.dtype}")
        print(f"  min: {output.min().item():.6f}, max: {output.max().item():.6f}, mean: {output.mean().item():.6f}")
        if torch.all((output == 0) | (output == 1)):
            frac_ones = (output == 1).float().mean().item()
            print(f"  fraction_ones: {frac_ones:.4f}")
            print("  ✓ Output is BINARY (only 0/1)")
        else:
            print(f"  ✗ Output is NOT binary (has intermediate values)")
        print("="*80 + "\n")
        
        return output
    return wrapper


def patch_sliding_window():
    """Patch _sliding_window_predict."""
    from src.archs import flow_model_vlm_film
    global original_sliding_window_predict
    
    # Find the class
    FlowModelVLMFiLM = flow_model_vlm_film.FlowModelVLMFiLM
    original_sliding_window_predict = FlowModelVLMFiLM._sliding_window_predict
    FlowModelVLMFiLM._sliding_window_predict = instrumented_sliding_window_predict(original_sliding_window_predict)


def patch_test_step():
    """Patch _test_step_dfm to dump preds."""
    from src.archs import flow_model_vlm_film
    FlowModelVLMFiLM = flow_model_vlm_film.FlowModelVLMFiLM
    
    original_test_step = FlowModelVLMFiLM._test_step_dfm
    
    def instrumented_test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()
        
        self._maybe_save_intermediate_eval(images, sample_names)
        
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                if getattr(self.hparams, 'use_sliding_infer', True):
                    output_geometry = self._sliding_window_predict(images)
                else:
                    output_geometry = self.sample(None, images)
                output_geometry_list.append(output_geometry)
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            if getattr(self.hparams, 'use_sliding_infer', True):
                output_geometry = self._sliding_window_predict(images)
            else:
                output_geometry = self.sample(None, images)
        
        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)
        
        preds = (output_geometry > 0.5).long()
        
        # === DUMP PREDS ===
        debug_dir = Path("debug_dumps")
        debug_dir.mkdir(exist_ok=True)
        torch.save(preds[0].cpu(), debug_dir / "debug_preds.pt")
        
        print("\n" + "="*80)
        print("FINAL PREDICTIONS (after threshold)")
        print("="*80)
        print(f"preds shape: {preds.shape}, dtype: {preds.dtype}")
        print(f"  unique values: {preds.unique().cpu().tolist()}")
        frac_ones = (preds == 1).float().mean().item()
        print(f"  fraction_ones: {frac_ones:.4f}")
        print("="*80 + "\n")
        
        # Continue with original metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()}, sync_dist=True)
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()}, sync_dist=True)
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='test')
        
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks_binary = (preds > 0).float()
            label_masks = (labels > 0).float()
    
    FlowModelVLMFiLM._test_step_dfm = instrumented_test_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    print("="*80)
    print("DFM BINARY SEGMENTATION DEBUGGER")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GPU: {args.gpu}")
    print("\nThis script will:")
    print("  1. Run inference on ONE batch")
    print("  2. Dump intermediate tensors to debug_dumps/")
    print("  3. Print detailed statistics")
    print("  4. Help diagnose fragmentation issue")
    print("="*80 + "\n")
    
    # Patch before import
    patch_sampler()
    patch_sliding_window()
    patch_test_step()
    
    # Now run evaluation with standard script
    print("Patching complete. Now run:")
    print(f"  uv run python scripts/evaluate_vlm_film.py --checkpoint {args.checkpoint} --gpu {args.gpu} --data xca --models medsegdiff_flow")
    print("\nOr integrate this into your eval script.\n")
    
    unpatch_sampler()


if __name__ == '__main__':
    main()
