#!/usr/bin/env python
"""
Analyze Betti-1 error patterns between baseline and VLM-FiLM experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage

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


def main():
    # Load sample metrics
    exp1_metrics = pd.read_csv('/home/yongjun/sofl-seg/results/evaluation/medsegdiff_flow/medsegdiff_flow_xca_20260128_105314/predictions/sample_metrics.csv')
    exp2_metrics = pd.read_csv('/home/yongjun/sofl-seg/results/evaluation/medsegdiff_flow/medsegdiff_flow_xca_20260129_174302/predictions/sample_metrics.csv')
    
    # Convert numeric columns
    numeric_cols = ['dice', 'iou', 'precision', 'recall', 'specificity', 'cldice', 
                    'connectivity', 'density_error', 'betti_0_error', 'betti_1_error']
    for col in numeric_cols:
        exp1_metrics[col] = pd.to_numeric(exp1_metrics[col], errors='coerce')
        exp2_metrics[col] = pd.to_numeric(exp2_metrics[col], errors='coerce')
    
    # Merge on sample name
    df = exp1_metrics.merge(exp2_metrics, on='sample_name', suffixes=('_baseline', '_vlm'))
    
    # Compute differences
    df['betti_1_error_diff'] = df['betti_1_error_vlm'] - df['betti_1_error_baseline']
    df['dice_diff'] = df['dice_vlm'] - df['dice_baseline']
    df['cldice_diff'] = df['cldice_vlm'] - df['cldice_baseline']
    
    print("=" * 80)
    print("Betti-1 Error Analysis: Baseline vs VLM-FiLM")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print()
    
    # Summary statistics
    print("Summary Statistics")
    print("-" * 80)
    print(f"Betti-1 Error:")
    print(f"  Baseline:  {df['betti_1_error_baseline'].mean():.2f} ± {df['betti_1_error_baseline'].std():.2f}")
    print(f"  VLM-FiLM:  {df['betti_1_error_vlm'].mean():.2f} ± {df['betti_1_error_vlm'].std():.2f}")
    print(f"  Diff:      {df['betti_1_error_diff'].mean():.2f} ± {df['betti_1_error_diff'].std():.2f}")
    print()
    print(f"Dice Score:")
    print(f"  Baseline:  {df['dice_baseline'].mean():.4f} ± {df['dice_baseline'].std():.4f}")
    print(f"  VLM-FiLM:  {df['dice_vlm'].mean():.4f} ± {df['dice_vlm'].std():.4f}")
    print(f"  Diff:      {df['dice_diff'].mean():.4f} ± {df['dice_diff'].std():.4f}")
    print()
    print(f"clDice Score:")
    print(f"  Baseline:  {df['cldice_baseline'].mean():.4f} ± {df['cldice_baseline'].std():.4f}")
    print(f"  VLM-FiLM:  {df['cldice_vlm'].mean():.4f} ± {df['cldice_vlm'].std():.4f}")
    print(f"  Diff:      {df['cldice_diff'].mean():.4f} ± {df['cldice_diff'].std():.4f}")
    print()
    
    # Samples with increased Betti-1 error
    increased = df[df['betti_1_error_diff'] > 0].sort_values('betti_1_error_diff', ascending=False)
    decreased = df[df['betti_1_error_diff'] < 0].sort_values('betti_1_error_diff')
    no_change = df[df['betti_1_error_diff'] == 0]
    
    print("=" * 80)
    print(f"Betti-1 Error Change Distribution")
    print("=" * 80)
    print(f"Increased: {len(increased)} samples ({len(increased)/len(df)*100:.1f}%)")
    print(f"Decreased: {len(decreased)} samples ({len(decreased)/len(df)*100:.1f}%)")
    print(f"No change: {len(no_change)} samples ({len(no_change)/len(df)*100:.1f}%)")
    print()
    
    # Top samples with increased error
    if len(increased) > 0:
        print("=" * 80)
        print(f"Top 15 Samples with INCREASED Betti-1 Error")
        print("=" * 80)
        top_cols = ['sample_name', 'betti_1_error_baseline', 'betti_1_error_vlm', 'betti_1_error_diff', 
                    'dice_baseline', 'dice_vlm', 'dice_diff', 'cldice_diff']
        print(increased[top_cols].head(15).to_string(index=False))
        print()
        
        # Analyze FP patterns for top samples
        print("=" * 80)
        print("False Positive Pattern Analysis (Top 10 Increased Error Samples)")
        print("=" * 80)
        
        exp2_pred_dir = Path('/home/yongjun/sofl-seg/results/evaluation/medsegdiff_flow/medsegdiff_flow_xca_20260129_174302/predictions')
        
        fp_results = []
        for idx, row in increased.head(10).iterrows():
            sample_name = row['sample_name']
            
            pred_path = exp2_pred_dir / sample_name / 'prediction.png'
            gt_path = exp2_pred_dir / sample_name / 'label.png'
            img_path = exp2_pred_dir / sample_name / 'image.png'
            
            if not all([pred_path.exists(), gt_path.exists(), img_path.exists()]):
                continue
            
            pred = np.array(Image.open(pred_path)) / 255.0
            gt = np.array(Image.open(gt_path)) / 255.0
            image = np.array(Image.open(img_path))
            
            fp_pattern = classify_fp_pattern(pred, gt, image)
            fp_pattern['sample_name'] = sample_name
            fp_pattern['betti_1_error_diff'] = row['betti_1_error_diff']
            fp_results.append(fp_pattern)
        
        if fp_results:
            fp_df = pd.DataFrame(fp_results)
            
            print("\nPer-sample FP Pattern Breakdown:")
            print(fp_df[['sample_name', 'betti_1_error_diff', 'junction', 'background', 'artifact', 'total_fp']].to_string(index=False))
            
            print("\n" + "=" * 80)
            print("Aggregated FP Pattern Distribution")
            print("=" * 80)
            total_fp = fp_df['total_fp'].sum()
            total_junction = fp_df['junction'].sum()
            total_background = fp_df['background'].sum()
            total_artifact = fp_df['artifact'].sum()
            
            print(f"\nTotal FP pixels (top 10 samples): {total_fp:,}")
            print(f"  Junction FP:    {total_junction:,} ({total_junction/total_fp*100:.1f}%)")
            print(f"  Background FP:  {total_background:,} ({total_background/total_fp*100:.1f}%)")
            print(f"  Artifact FP:    {total_artifact:,} ({total_artifact/total_fp*100:.1f}%)")
            
            print(f"\nAverage per sample:")
            print(f"  Junction FP:    {fp_df['junction'].mean():.0f} pixels")
            print(f"  Background FP:  {fp_df['background'].mean():.0f} pixels")
            print(f"  Artifact FP:    {fp_df['artifact'].mean():.0f} pixels")
    
    # Top samples with decreased error
    if len(decreased) > 0:
        print("\n" + "=" * 80)
        print(f"Top 10 Samples with DECREASED Betti-1 Error")
        print("=" * 80)
        top_cols = ['sample_name', 'betti_1_error_baseline', 'betti_1_error_vlm', 'betti_1_error_diff', 
                    'dice_diff', 'cldice_diff']
        print(decreased[top_cols].head(10).to_string(index=False))
        print()
    
    # Correlation analysis
    print("=" * 80)
    print("Correlation Analysis")
    print("=" * 80)
    print(f"Betti-1 error diff vs Dice diff:   {df['betti_1_error_diff'].corr(df['dice_diff']):.3f}")
    print(f"Betti-1 error diff vs clDice diff: {df['betti_1_error_diff'].corr(df['cldice_diff']):.3f}")
    print()
    
    # Save detailed results
    output_path = '/home/yongjun/sofl-seg/results/evaluation/betti_error_comparison_detailed.csv'
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")


if __name__ == '__main__':
    main()
