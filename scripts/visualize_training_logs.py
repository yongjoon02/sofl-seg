"""
Visualize training logs for all XCA models.
"""

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import pandas as pd


def load_tensorboard_logs(log_dir: Path) -> dict:
    """Load metrics from TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'step': [e.step for e in events],
            'value': [e.value for e in events],
        }
    return data


def main():
    # Find all XCA experiment logs
    experiments_dir = Path("experiments")
    models = ['csnet', 'dscnet', 'medsegdiff', 'berdiff']
    colors = {'csnet': '#1f77b4', 'dscnet': '#ff7f0e', 'medsegdiff': '#2ca02c', 'berdiff': '#d62728'}
    
    # Load logs
    all_logs = {}
    for model in models:
        log_dirs = list(experiments_dir.glob(f"{model}/xca/{model}_xca_*/tensorboard"))
        if log_dirs:
            log_dir = sorted(log_dirs)[-1]  # Latest run
            try:
                all_logs[model] = load_tensorboard_logs(log_dir)
                print(f"✅ Loaded {model}: {log_dir}")
            except Exception as e:
                print(f"❌ Failed to load {model}: {e}")
    
    if not all_logs:
        print("No logs found!")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('XCA Training Progress - All Models', fontsize=14, fontweight='bold')
    
    # Metrics to plot
    metrics = [
        ('train/loss', 'Training Loss', axes[0, 0]),
        ('val/dice', 'Validation Dice', axes[0, 1]),
        ('val/iou', 'Validation IoU', axes[1, 0]),
        ('val/recall', 'Validation Recall', axes[1, 1]),
    ]
    
    for metric_key, metric_name, ax in metrics:
        for model, logs in all_logs.items():
            if metric_key in logs:
                steps = logs[metric_key]['step']
                values = logs[metric_key]['value']
                ax.plot(steps, values, label=model, color=colors[model], alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("results/xca/training_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved training curves to: {output_path}")
    
    # Also create a summary table
    print("\n" + "="*60)
    print("Final Validation Metrics (Last Logged Value)")
    print("="*60)
    
    summary_data = []
    for model, logs in all_logs.items():
        row = {'Model': model}
        for metric in ['val/dice', 'val/iou', 'val/precision', 'val/recall']:
            if metric in logs and logs[metric]['value']:
                row[metric.split('/')[1]] = logs[metric]['value'][-1]
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    plt.show()


if __name__ == "__main__":
    main()
