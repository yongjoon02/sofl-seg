"""
Analyze cosine similarity distribution of conditioning vectors within training batches.

This script loads the actual training dataset and VLM model to compute 
conditioning vectors for real batches, then analyzes their similarity.
"""
import autorootcwd
import torch
import numpy as np
from pathlib import Path
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
import sys
import yaml

# Add src to path
sys.path.insert(0, '/home/yongjun/sofl-seg')

from src.data.xca import XCADataset
from src.conditioning.vlm_film_conditioner import VLMFiLMConditioner

def load_config(config_path):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_batch_similarities(dataloader, vlm_model, num_batches=50, device='cuda'):
    """
    Compute pairwise cosine similarities within real training batches.
    
    Args:
        dataloader: DataLoader for the dataset
        vlm_model: VLM conditioning model
        num_batches: Number of batches to analyze
        device: Device to use for computation
    
    Returns:
        all_similarities: List of similarity values from all batches
        batch_stats: Statistics for each batch
        cond_vecs_all: All conditioning vectors (optional, for further analysis)
    """
    vlm_model.eval()
    vlm_model.to(device)
    
    all_similarities = []
    batch_stats = []
    cond_vecs_all = []
    
    print(f"\nComputing similarities for {num_batches} batches")
    print(f"Batch size: {dataloader.batch_size}")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Get images
            images = batch['image'].to(device)
            batch_size = images.shape[0]
            
            # Generate conditioning vectors via VLM
            # VLMConditioner returns dict with 'cond_vec', 'film_gamma', 'film_beta'
            cond_dict = vlm_model(images)
            cond_vecs = cond_dict['cond_vec']  # Shape: (B, hidden_dim)
            
            # Aggregate if needed (spatial dimensions)
            if cond_vecs.dim() > 2:
                # Pool over spatial dimensions: (B, C, H, W) -> (B, C)
                cond_vecs = cond_vecs.flatten(2).mean(dim=2)
            elif cond_vecs.dim() == 2 and cond_vecs.shape[0] != batch_size:
                # Might be (num_patches, hidden_dim) - reshape
                # This shouldn't happen with proper batching, but just in case
                print(f"Warning: Unexpected shape {cond_vecs.shape} for batch size {batch_size}")
                continue
            
            # Move to CPU for computation
            cond_vecs_cpu = cond_vecs.cpu()
            cond_vecs_all.append(cond_vecs_cpu)
            
            # Normalize vectors
            cond_vecs_norm = cond_vecs_cpu / cond_vecs_cpu.norm(dim=1, keepdim=True)
            
            # Compute similarity matrix
            sim_matrix = cond_vecs_norm @ cond_vecs_norm.T
            
            # Extract upper triangular part (excluding diagonal)
            triu_indices = torch.triu_indices(batch_size, batch_size, offset=1)
            batch_sims = sim_matrix[triu_indices[0], triu_indices[1]].numpy()
            
            all_similarities.extend(batch_sims)
            
            # Batch statistics
            batch_stats.append({
                'batch_idx': i,
                'mean': batch_sims.mean(),
                'std': batch_sims.std(),
                'min': batch_sims.min(),
                'max': batch_sims.max(),
                'median': np.median(batch_sims)
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_batches} batches...")
    
    return all_similarities, batch_stats, torch.cat(cond_vecs_all, dim=0)

def print_analysis_summary(similarities, batch_stats):
    """Print detailed analysis summary."""
    
    print("\n" + "="*80)
    print("CONDITIONING VECTOR SIMILARITY ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total pairwise comparisons: {len(similarities)}")
    print(f"  Number of batches analyzed: {len(batch_stats)}")
    
    print(f"\n📈 Similarity Distribution:")
    print(f"  Mean:   {np.mean(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  Std:    {np.std(similarities):.4f}")
    print(f"  Min:    {np.min(similarities):.4f}")
    print(f"  Max:    {np.max(similarities):.4f}")
    print(f"  Q1:     {np.percentile(similarities, 25):.4f}")
    print(f"  Q3:     {np.percentile(similarities, 75):.4f}")
    print(f"  IQR:    {np.percentile(similarities, 75) - np.percentile(similarities, 25):.4f}")
    
    # Interpretation
    mean_sim = np.mean(similarities)
    print(f"\n🎯 Interpretation:")
    
    if mean_sim > 0.9:
        print("  ⚠️  VERY HIGH similarity (>0.9)")
        print("     → Conditioning vectors are highly similar within batches")
        print("     → Limited diversity in visual features")
        print("     → May indicate overfitting to common patterns")
        print("     → VLM might be extracting only global/common features")
    elif mean_sim > 0.7:
        print("  ⚠️  HIGH similarity (0.7-0.9)")
        print("     → Conditioning vectors show moderate-high similarity")
        print("     → Some common features but limited diversity")
        print("     → VLM captures shared characteristics across samples")
        print("     → Consider: Are samples in batch very similar (e.g., same patient)?")
    elif mean_sim > 0.5:
        print("  ✅ MODERATE similarity (0.5-0.7)")
        print("     → Good balance between similarity and diversity")
        print("     → VLM captures both shared and unique features")
        print("     → Healthy conditioning vector distribution")
        print("     → FiLM conditioning should provide useful modulation")
    elif mean_sim > 0.3:
        print("  ✅ LOW similarity (0.3-0.5)")
        print("     → High diversity in conditioning vectors")
        print("     → VLM captures distinct features per sample")
        print("     → Good discrimination capability")
        print("     → Strong sample-specific conditioning")
    else:
        print("  ⚠️  VERY LOW similarity (<0.3)")
        print("     → Conditioning vectors are very different")
        print("     → May indicate high noise or very diverse dataset")
        print("     → Or: VLM might be unstable/overfitting to noise")
        print("     → Check if VLM conditioning is working properly")
    
    # Batch consistency
    batch_means = [stat['mean'] for stat in batch_stats]
    batch_std = np.std(batch_means)
    
    print(f"\n📦 Batch Consistency:")
    print(f"  Mean similarity across batches: {np.mean(batch_means):.4f}")
    print(f"  Std of batch means: {batch_std:.4f}")
    
    if batch_std < 0.05:
        print("  ✅ Very consistent across batches (low variance)")
        print("     → Batch composition doesn't strongly affect similarity")
    elif batch_std < 0.1:
        print("  ✅ Reasonably consistent across batches")
    else:
        print("  ⚠️  High variance across batches")
        print("     → Some batches much more similar/diverse than others")
        print("     → Batch sampling might create uneven difficulty")
    
    # Most similar and dissimilar batches
    most_similar_batch = max(batch_stats, key=lambda x: x['mean'])
    most_diverse_batch = min(batch_stats, key=lambda x: x['mean'])
    
    print(f"\n  Most similar batch (#{most_similar_batch['batch_idx']}):")
    print(f"    Mean similarity: {most_similar_batch['mean']:.4f}")
    print(f"    → Samples in this batch are very similar")
    
    print(f"\n  Most diverse batch (#{most_diverse_batch['batch_idx']}):")
    print(f"    Mean similarity: {most_diverse_batch['mean']:.4f}")
    print(f"    → Samples in this batch are quite different")
    
    # Distribution shape
    print(f"\n📊 Distribution Shape:")
    skewness = np.mean([(x - mean_sim)**3 for x in similarities]) / (np.std(similarities)**3)
    kurtosis = np.mean([(x - mean_sim)**4 for x in similarities]) / (np.std(similarities)**4) - 3
    
    print(f"  Skewness: {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("    → Roughly symmetric distribution")
    elif skewness > 0:
        print("    → Right-skewed (more low similarities)")
    else:
        print("    → Left-skewed (more high similarities)")
    
    print(f"  Excess Kurtosis: {kurtosis:.4f}")
    if abs(kurtosis) < 0.5:
        print("    → Normal-like distribution (mesokurtic)")
    elif kurtosis > 0:
        print("    → Heavy tails (leptokurtic) - some extreme values")
    else:
        print("    → Light tails (platykurtic) - concentrated around mean")
    
    print("\n" + "="*80)

def main():
    print("🔍 Analyzing Conditioning Vector Similarity in Real Training Batches")
    print("="*80)
    
    # Configuration
    config_path = "/home/yongjun/sofl-seg/experiments/medsegdiff_flow/xca/medsegdiff_flow_xca_20260205_015359/config.yaml"
    num_batches = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    # Load config
    print("\n📂 Loading configuration...")
    config = load_config(config_path)
    
    # Create dataset
    print("\n📂 Loading dataset...")
    dataset = XCADataset(
        path=config['data']['train_dir'],
        crop_size=config['data']['image_size'],
        augmentation=False  # No augmentation for analysis
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['train_bs'],
        shuffle=True,
        num_workers=0,  # Single process for simplicity
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"  Batch size: {config['data']['train_bs']}")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Initialize VLM model
    print("\n🤖 Initializing VLM conditioning model...")
    vlm_config = config['model']['vlm_film_config']
    vlm_model = VLMFiLMConditioner(
        enabled=True,
        model_name=vlm_config['model_name'],
        cond_dim=vlm_config['cond_dim'],
        cache_dir=vlm_config.get('cache_dir', 'cache/vlm_profiles'),
        verbose=True
    )
    
    print(f"  VLM model: {vlm_config['model_name']}")
    print(f"  Cond dim: {vlm_config['cond_dim']}")
    
    # Compute similarities
    print("\n⚙️  Computing conditioning vectors and similarities...")
    try:
        similarities, batch_stats, cond_vecs = compute_batch_similarities(
            dataloader,
            vlm_model,
            num_batches=num_batches,
            device=device
        )
    except Exception as e:
        print(f"❌ Error computing similarities: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print analysis
    print_analysis_summary(similarities, batch_stats)
    
    # Additional statistics
    print("\n📐 Conditioning Vector Statistics:")
    print(f"  Total vectors analyzed: {cond_vecs.shape[0]}")
    print(f"  Vector dimension: {cond_vecs.shape[1]}")
    print(f"  Mean norm: {cond_vecs.norm(dim=1).mean():.4f} ± {cond_vecs.norm(dim=1).std():.4f}")
    print(f"  Min norm: {cond_vecs.norm(dim=1).min():.4f}")
    print(f"  Max norm: {cond_vecs.norm(dim=1).max():.4f}")
    
    # Save results
    output_file = "cond_vec_similarity_stats.txt"
    with open(output_file, 'w') as f:
        f.write("CONDITIONING VECTOR SIMILARITY ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total comparisons: {len(similarities)}\n")
        f.write(f"Batches analyzed: {len(batch_stats)}\n\n")
        f.write("Similarity Statistics:\n")
        f.write(f"  Mean: {np.mean(similarities):.4f}\n")
        f.write(f"  Std: {np.std(similarities):.4f}\n")
        f.write(f"  Min: {np.min(similarities):.4f}\n")
        f.write(f"  Max: {np.max(similarities):.4f}\n")
        f.write(f"  Median: {np.median(similarities):.4f}\n")
        f.write(f"  Q1: {np.percentile(similarities, 25):.4f}\n")
        f.write(f"  Q3: {np.percentile(similarities, 75):.4f}\n")
    
    print(f"\n✅ Saved statistics to {output_file}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
