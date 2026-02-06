#!/usr/bin/env python
"""
Compare Betti-1 errors between two experiments and classify FP patterns.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import pandas as pd
from skimage import measure
from scipy import ndimage

def compute_betti_numbers(mask):
    """Compute Betti-0 and Betti-1 numbers"""
    if mask.max() == 0:
        return 0, 0
    
    # Betti-0: connected components
    labeled = measure.label(mask > 0.5, connectivity=2)
    betti_0 = labeled.max()
    
    # Betti-1: Euler characteristic method
    # χ = V - E + F = #components - #holes
    # For 2D: Betti-1 = V - E + F - Betti-0
    # Approximate using morphological operations
    
    # Fill holes to count them
    filled = ndimage.binary_fill_holes(mask > 0.5)
    holes = filled.astype(int) - (mask > 0.5).astype(int)
    labeled_holes = measure.label(holes, connectivity=2)
    betti_1 = labeled_holes.max()
    
    return betti_0, betti_1


def classify_fp_pattern(pred, gt, image):
    """
    Classify False Positive patterns:
    - junction: FP near vessel junctions
    - background: FP in background clutter
    - artifact: FP on catheter/artifact
    """
    # False Positives
    fp_mask = (pred > 0.5) & (gt < 0.5)
    
    if fp_mask.sum() == 0:
        return {'junction': 0, 'background': 0, 'artifact': 0, 'total_fp': 0}
    
    # Dilate GT to find junction regions
    kernel = np.ones((5, 5), np.uint8)
    gt_dilated = cv2.dilate((gt > 0.5).astype(np.uint8), kernel, iterations=2)
    gt_eroded = cv2.erode((gt > 0.5).astype(np.uint8), kernel, iterations=1)
    junction_region = (gt_dilated - gt_eroded) > 0
    
    # Classify FP pixels
    fp_junction = fp_mask & junction_region
    fp_artifact = fp_mask & (image > 200)  # Bright artifacts (catheter)
    fp_background = fp_mask & ~fp_junction & ~fp_artifact
    
    return {
        'junction': fp_junction.sum(),
        'background': fp_background.sum(),
        'artifact': fp_artifact.sum(),
        'total_fp': fp_mask.sum()
    }


def analyze_experiments(exp1_dir, exp2_dir, data_dir):
    """Compare two experiments and analyze Betti-1 error increases"""
    
    exp1_pred_dir = Path(exp1_dir) / 'predictions'
    exp2_pred_dir = Path(exp2_dir) / 'predictions'
    
    results = []
    
    # Get all sample IDs
    sample_ids = sorted([d.name for d in exp1_pred_dir.iterdir() if d.is_dir()])
    
    print(f"Analyzing {len(sample_ids)} samples...")
    
    for sample_id in sample_ids:
        # Load predictions (already in prediction directory)
        pred1_path = exp1_pred_dir / sample_id / 'prediction.png'
        pred2_path = exp2_pred_dir / sample_id / 'prediction.png'
        gt_path = exp1_pred_dir / sample_id / 'label.png'  # GT is copied to prediction dir
        img_path = exp1_pred_dir / sample_id / 'image.png'  # Image is copied to prediction dir
        
        if not all([pred1_path.exists(), pred2_path.exists(), gt_path.exists(), img_path.exists()]):
            print(f"Skipping {sample_id}: missing files")
            continue
        
        # Load images
        pred1 = np.array(Image.open(pred1_path)) / 255.0
        pred2 = np.array(Image.open(pred2_path)) / 255.0
        gt = np.array(Image.open(gt_path)) / 255.0
        image = np.array(Image.open(img_path))
        
        # Compute Betti numbers
        betti_0_pred1, betti_1_pred1 = compute_betti_numbers(pred1)
        betti_0_pred2, betti_1_pred2 = compute_betti_numbers(pred2)
        betti_0_gt, betti_1_gt = compute_betti_numbers(gt)
        
        # Betti-1 errors
        betti_1_error_exp1 = abs(betti_1_pred1 - betti_1_gt)
        betti_1_error_exp2 = abs(betti_1_pred2 - betti_1_gt)
        betti_1_error_diff = betti_1_error_exp2 - betti_1_error_exp1
        
        # Classify FP patterns for exp2
        fp_pattern = classify_fp_pattern(pred2, gt, image)
        
        # Compute basic metrics
        dice1 = 2 * (pred1 * gt).sum() / (pred1.sum() + gt.sum() + 1e-8)
        dice2 = 2 * (pred2 * gt).sum() / (pred2.sum() + gt.sum() + 1e-8)
        
        results.append({
            'sample_id': sample_id,
            'betti_1_gt': betti_1_gt,
            'betti_1_exp1': betti_1_pred1,
            'betti_1_exp2': betti_1_pred2,
            'betti_1_error_exp1': betti_1_error_exp1,
            'betti_1_error_exp2': betti_1_error_exp2,
            'betti_1_error_diff': betti_1_error_diff,
            'dice_exp1': dice1,
            'dice_exp2': dice2,
            'dice_diff': dice2 - dice1,
            'fp_junction': fp_pattern['junction'],
            'fp_background': fp_pattern['background'],
            'fp_artifact': fp_pattern['artifact'],
            'fp_total': fp_pattern['total_fp'],
        })
    
    df = pd.DataFrame(results)
    return df


def main():
    exp1_dir = '/home/yongjun/sofl-seg/results/evaluation/medsegdiff_flow/medsegdiff_flow_xca_20260128_105314'
    exp2_dir = '/home/yongjun/sofl-seg/results/evaluation/medsegdiff_flow/medsegdiff_flow_xca_20260129_174302'
    data_dir = '/home/yongjun/sofl-seg/data/xca_full'
    
    print("=" * 70)
    print("Comparing Betti-1 Errors Between Experiments")
    print("=" * 70)
    print(f"Experiment 1 (Baseline): {exp1_dir.split('/')[-1]}")
    print(f"Experiment 2 (VLM-FiLM): {exp2_dir.split('/')[-1]}")
    print()
    
    df = analyze_experiments(exp1_dir, exp2_dir, data_dir)
    
    # Save full results
    output_path = 'results/evaluation/betti_error_comparison.csv'
    df.to_csv(output_path, index=False)
    print(f"Full results saved to: {output_path}\n")
    
    # Summary statistics
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print(f"\nBetti-1 Error:")
    print(f"  Exp1 (Baseline): {df['betti_1_error_exp1'].mean():.2f} ± {df['betti_1_error_exp1'].std():.2f}")
    print(f"  Exp2 (VLM-FiLM): {df['betti_1_error_exp2'].mean():.2f} ± {df['betti_1_error_exp2'].std():.2f}")
    print(f"  Difference: {df['betti_1_error_diff'].mean():.2f} ± {df['betti_1_error_diff'].std():.2f}")
    
    print(f"\nDice Score:")
    print(f"  Exp1 (Baseline): {df['dice_exp1'].mean():.4f} ± {df['dice_exp1'].std():.4f}")
    print(f"  Exp2 (VLM-FiLM): {df['dice_exp2'].mean():.4f} ± {df['dice_exp2'].std():.4f}")
    print(f"  Difference: {df['dice_diff'].mean():.4f} ± {df['dice_diff'].std():.4f}")
    
    # Samples with increased Betti-1 error
    increased = df[df['betti_1_error_diff'] > 0].sort_values('betti_1_error_diff', ascending=False)
    
    print("\n" + "=" * 70)
    print(f"Samples with INCREASED Betti-1 Error (n={len(increased)})")
    print("=" * 70)
    
    if len(increased) > 0:
        print("\nTop 10 samples with largest increase:")
        print(increased[['sample_id', 'betti_1_error_diff', 'dice_diff', 
                        'fp_junction', 'fp_background', 'fp_artifact']].head(10).to_string(index=False))
        
        # FP pattern analysis for increased error samples
        print("\n" + "=" * 70)
        print("False Positive Pattern Analysis (Increased Error Samples)")
        print("=" * 70)
        
        total_fp = increased['fp_total'].sum()
        fp_junction = increased['fp_junction'].sum()
        fp_background = increased['fp_background'].sum()
        fp_artifact = increased['fp_artifact'].sum()
        
        print(f"\nTotal FP pixels: {total_fp:,}")
        print(f"  Junction FP:    {fp_junction:,} ({fp_junction/total_fp*100:.1f}%)")
        print(f"  Background FP:  {fp_background:,} ({fp_background/total_fp*100:.1f}%)")
        print(f"  Artifact FP:    {fp_artifact:,} ({fp_artifact/total_fp*100:.1f}%)")
        
        # Pattern distribution
        print("\n" + "=" * 70)
        print("Pattern Distribution per Sample (Average)")
        print("=" * 70)
        print(f"  Junction FP:    {increased['fp_junction'].mean():.1f} pixels/sample")
        print(f"  Background FP:  {increased['fp_background'].mean():.1f} pixels/sample")
        print(f"  Artifact FP:    {increased['fp_artifact'].mean():.1f} pixels/sample")
    
    # Samples with decreased Betti-1 error
    decreased = df[df['betti_1_error_diff'] < 0].sort_values('betti_1_error_diff')
    
    print("\n" + "=" * 70)
    print(f"Samples with DECREASED Betti-1 Error (n={len(decreased)})")
    print("=" * 70)
    
    if len(decreased) > 0:
        print("\nTop 10 samples with largest decrease:")
        print(decreased[['sample_id', 'betti_1_error_diff', 'dice_diff']].head(10).to_string(index=False))
    
    # No change
    no_change = df[df['betti_1_error_diff'] == 0]
    print(f"\nSamples with NO CHANGE: {len(no_change)}")
    
    print("\n" + "=" * 70)
    print("Correlation Analysis")
    print("=" * 70)
    print(f"Correlation between Betti-1 error change and Dice change: {df['betti_1_error_diff'].corr(df['dice_diff']):.3f}")
    
    return df


if __name__ == '__main__':
    main()
