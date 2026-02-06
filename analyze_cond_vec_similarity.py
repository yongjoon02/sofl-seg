"""
Analyze cosine similarity distribution of conditioning vectors within batches.

This script loads cached VLM profiles and computes pairwise cosine similarity
within randomly sampled batches to understand conditioning vector diversity.
"""

import torch
import numpy as np
from pathlib import Path
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib/seaborn not available. Skipping visualization.")

def load_vlm_profiles(cache_dir: Path):
    """Load all cached VLM conditioning vectors."""
    print(f"Loading VLM profiles from {cache_dir}")
    
    profiles = []
    profile_files = sorted(cache_dir.glob("*.pt"))
    
    if len(profile_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {cache_dir}")
    
    print(f"Found {len(profile_files)} cached profiles")
    
    for profile_file in tqdm(profile_files, desc="Loading profiles"):
        try:
            profile = torch.load(profile_file, map_location='cpu')
            # Profile might be dict with 'cond_vec' key or directly the tensor
            if isinstance(profile, dict):
                if 'cond_vec' in profile:
                    cond_vec = profile['cond_vec']
                elif 'conditioning' in profile:
                    cond_vec = profile['conditioning']
                else:
                    # Try to find any tensor in the dict
                    cond_vec = next(v for v in profile.values() if isinstance(v, torch.Tensor))
            else:
                cond_vec = profile
            
            # VLM profiles are typically (num_patches, hidden_dim)
            # We need to aggregate them to a single vector per image
            if cond_vec.dim() == 2:
                # Use mean pooling across patches
                cond_vec = cond_vec.mean(dim=0)
            elif cond_vec.dim() > 2:
                # Flatten and then mean pool
                cond_vec = cond_vec.flatten(0, -2).mean(dim=0)
            
            profiles.append(cond_vec)
        except Exception as e:
            print(f"Error loading {profile_file.name}: {e}")
            continue
    
    if len(profiles) == 0:
        raise ValueError("No valid profiles loaded")
    
    print(f"Successfully loaded {len(profiles)} conditioning vectors")
    print(f"Vector dimension: {profiles[0].shape}")
    
    return torch.stack(profiles)

def compute_batch_similarities(cond_vecs, batch_size=6, num_batches=100):
    """
    Sample random batches and compute pairwise cosine similarities within each batch.
    
    Args:
        cond_vecs: Tensor of shape (N, D) containing N conditioning vectors of dimension D
        batch_size: Number of samples per batch
        num_batches: Number of random batches to sample
    
    Returns:
        all_similarities: List of similarity values from all batches
        batch_stats: Statistics for each batch
    """
    N = cond_vecs.shape[0]
    all_similarities = []
    batch_stats = []
    
    print(f"\nComputing similarities for {num_batches} random batches of size {batch_size}")
    
    for i in tqdm(range(num_batches), desc="Sampling batches"):
        # Sample random batch
        indices = torch.randperm(N)[:batch_size]
        batch = cond_vecs[indices]
        
        # Compute pairwise cosine similarities
        # Normalize vectors
        batch_normalized = batch / batch.norm(dim=1, keepdim=True)
        
        # Compute similarity matrix
        sim_matrix = batch_normalized @ batch_normalized.T
        
        # Extract upper triangular part (excluding diagonal)
        triu_indices = torch.triu_indices(batch_size, batch_size, offset=1)
        batch_sims = sim_matrix[triu_indices[0], triu_indices[1]].cpu().numpy()
        
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
    
    return all_similarities, batch_stats

def plot_similarity_distribution(similarities, batch_stats, output_path="cond_vec_similarity_analysis.png"):
    """Create comprehensive visualization of similarity distribution."""
    
    if not PLOTTING_AVAILABLE:
        print("⚠️  Plotting libraries not available. Skipping visualization.")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histogram of all similarities
    ax1 = axes[0, 0]
    ax1.hist(similarities, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(similarities):.4f}')
    ax1.axvline(np.median(similarities), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(similarities):.4f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Pairwise Cosine Similarities Within Batches', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot(similarities, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Box Plot of Cosine Similarities', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    Mean: {np.mean(similarities):.4f}
    Median: {np.median(similarities):.4f}
    Std: {np.std(similarities):.4f}
    Min: {np.min(similarities):.4f}
    Max: {np.max(similarities):.4f}
    Q1: {np.percentile(similarities, 25):.4f}
    Q3: {np.percentile(similarities, 75):.4f}
    """
    ax2.text(1.5, np.median(similarities), stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Per-batch mean similarity
    ax3 = axes[1, 0]
    batch_means = [stat['mean'] for stat in batch_stats]
    ax3.plot(batch_means, marker='o', linestyle='-', alpha=0.6, markersize=4)
    ax3.axhline(np.mean(batch_means), color='red', linestyle='--', linewidth=2, 
                label=f'Overall Mean: {np.mean(batch_means):.4f}')
    ax3.set_xlabel('Batch Index', fontsize=12)
    ax3.set_ylabel('Mean Cosine Similarity', fontsize=12)
    ax3.set_title('Mean Similarity per Batch', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    ax4.plot(sorted_sims, cumulative, linewidth=2, color='steelblue')
    ax4.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(similarities):.4f}')
    ax4.axhline(0.5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Cosine Similarity', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    
    return fig

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
    elif mean_sim > 0.7:
        print("  ⚠️  HIGH similarity (0.7-0.9)")
        print("     → Conditioning vectors show moderate similarity")
        print("     → Some common features but reasonable diversity")
        print("     → VLM captures shared characteristics across samples")
    elif mean_sim > 0.5:
        print("  ✅ MODERATE similarity (0.5-0.7)")
        print("     → Good balance between similarity and diversity")
        print("     → VLM captures both shared and unique features")
        print("     → Healthy conditioning vector distribution")
    elif mean_sim > 0.3:
        print("  ✅ LOW similarity (0.3-0.5)")
        print("     → High diversity in conditioning vectors")
        print("     → VLM captures distinct features per sample")
        print("     → Good discrimination capability")
    else:
        print("  ⚠️  VERY LOW similarity (<0.3)")
        print("     → Conditioning vectors are very different")
        print("     → May indicate high noise or very diverse dataset")
        print("     → Check if VLM conditioning is working properly")
    
    # Batch consistency
    batch_means = [stat['mean'] for stat in batch_stats]
    batch_std = np.std(batch_means)
    
    print(f"\n📦 Batch Consistency:")
    print(f"  Mean similarity across batches: {np.mean(batch_means):.4f}")
    print(f"  Std of batch means: {batch_std:.4f}")
    
    if batch_std < 0.05:
        print("  ✅ Consistent across batches (low variance)")
    elif batch_std < 0.1:
        print("  ⚠️  Moderate variance across batches")
    else:
        print("  ⚠️  High variance across batches")
        print("     → Some batches much more similar than others")
    
    # Most similar and dissimilar batches
    most_similar_batch = max(batch_stats, key=lambda x: x['mean'])
    most_diverse_batch = min(batch_stats, key=lambda x: x['mean'])
    
    print(f"\n  Most similar batch (#{most_similar_batch['batch_idx']}):")
    print(f"    Mean similarity: {most_similar_batch['mean']:.4f}")
    print(f"  Most diverse batch (#{most_diverse_batch['batch_idx']}):")
    print(f"    Mean similarity: {most_diverse_batch['mean']:.4f}")
    
    print("\n" + "="*80)

def main():
    # Configuration
    cache_dir = Path("/home/yongjun/sofl-seg/cache/vlm_profiles")
    batch_size = 6  # Same as training batch size
    num_batches = 200  # Number of random batches to sample
    output_path = "cond_vec_similarity_analysis.png"
    
    print("🔍 Analyzing Conditioning Vector Similarity Distribution")
    print("="*80)
    
    # Load conditioning vectors
    try:
        cond_vecs = load_vlm_profiles(cache_dir)
        print(f"\nℹ️  Note: Found {len(cond_vecs)} cached profiles")
        
        # Adjust batch size and num_batches based on available data
        if len(cond_vecs) < batch_size:
            print(f"⚠️  Only {len(cond_vecs)} profiles available, but batch_size={batch_size}")
            print(f"   Adjusting batch_size to {len(cond_vecs)}")
            batch_size = len(cond_vecs)
        
        if len(cond_vecs) < 20:
            print(f"⚠️  Very few profiles available ({len(cond_vecs)})")
            print("   Consider running training to generate more cached profiles")
            print("   Proceeding with limited analysis...")
            num_batches = min(num_batches, 10)
            
    except Exception as e:
        print(f"❌ Error loading profiles: {e}")
        print("\n💡 This might be because:")
        print("   1. VLM profiles haven't been cached yet")
        print("   2. The cache directory is empty or has different structure")
        print("   3. Training needs to be run to generate VLM profiles")
        return
    
    # Compute similarities
    try:
        similarities, batch_stats = compute_batch_similarities(
            cond_vecs, 
            batch_size=batch_size, 
            num_batches=num_batches
        )
    except Exception as e:
        print(f"❌ Error computing similarities: {e}")
        return
    
    # Print analysis
    print_analysis_summary(similarities, batch_stats)
    
    # Create visualization
    if PLOTTING_AVAILABLE:
        try:
            plot_similarity_distribution(similarities, batch_stats, output_path)
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    else:
        print("\n⚠️  Skipping visualization (matplotlib/seaborn not installed)")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
