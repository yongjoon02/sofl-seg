"""Custom logger for saving predictions and metrics."""

import csv
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image


class PredictionLogger(Logger):
    """Logger that saves predictions, labels, and per-sample metrics."""

    def __init__(self, save_dir: str, name: str = "prediction", version: str = None):
        super().__init__()
        self._save_dir = Path(save_dir) / name
        if version is not None:
            self._save_dir = self._save_dir / version

        # Create directories
        self.pred_dir = self._save_dir
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self._save_dir / "sample_metrics.csv"

        # Initialize CSV file
        self._init_csv()

        self._name = name
        self._version = version

        # Store all metrics for computing average
        self._all_metrics = []

    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'sample_name', 'dice', 'iou', 'precision', 'recall',
                    'specificity', 'cldice', 'connectivity', 'density_error',
                    'betti_0_error', 'betti_1_error'
                ])

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def save_dir(self):
        return str(self._save_dir)

    @rank_zero_only
    def log_hyperparams(self, params):
        """Log hyperparameters."""
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """Log metrics."""
        pass

    @rank_zero_only
    def save_predictions(self, sample_names, images, predictions, labels, metrics_dict):
        """Save predictions, labels, and metrics for each sample."""
        for i, (sample_name, img, pred, label, metrics) in enumerate(zip(
            sample_names, images, predictions, labels, metrics_dict
        )):
            # Convert tensors to numpy
            if isinstance(img, torch.Tensor):
                img = img.squeeze().cpu().numpy()
            if isinstance(pred, torch.Tensor):
                pred = pred.squeeze().cpu().numpy()
            if isinstance(label, torch.Tensor):
                label = label.squeeze().cpu().numpy()

            # Extract filename from sample_name (remove path prefix)
            if '/' in sample_name:
                filename = sample_name.split('/')[-1]  # Get last part after /
                if '.' in filename:
                    filename = filename.rsplit('.', 1)[0]  # Remove extension
            else:
                filename = sample_name
                if '.' in filename:
                    filename = filename.rsplit('.', 1)[0]

            # Normalize image to 0-255 (handle negative values)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255
            img = np.clip(img, 0, 255).astype(np.uint8)

            # Fix axis orientation - transpose to match original
            img = img.T

            # Convert predictions and labels to 0-255
            pred = (pred * 255).astype(np.uint8)
            label = (label * 255).astype(np.uint8)

            # Fix axis orientation - transpose to match original
            pred = pred.T
            label = label.T

            # Save images
            sample_dir = self.pred_dir / filename
            sample_dir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(img, mode='L').save(sample_dir / "image.png")
            Image.fromarray(pred, mode='L').save(sample_dir / "prediction.png")
            Image.fromarray(label, mode='L').save(sample_dir / "label.png")

            # Save metrics to CSV
            self._save_sample_metrics(filename, metrics)

    def _save_sample_metrics(self, sample_name, metrics):
        """Save metrics for a single sample to CSV."""
        # Store metrics for averaging later
        self._all_metrics.append(metrics)

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                sample_name,
                metrics.get('dice', 0.0),
                metrics.get('iou', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('specificity', 0.0),
                metrics.get('cldice', 0.0),
                metrics.get('connectivity', 0.0),
                metrics.get('density_error', 0.0),
                metrics.get('betti_0_error', 0.0),
                metrics.get('betti_1_error', 0.0),
            ])

    @rank_zero_only
    def finalize(self, status):
        """Finalize logging - compute and append average metrics."""
        if len(self._all_metrics) > 0:
            # Compute averages
            metric_keys = ['dice', 'iou', 'precision', 'recall', 'specificity',
                          'cldice', 'connectivity', 'density_error',
                          'betti_0_error', 'betti_1_error']

            avg_metrics = {}
            for key in metric_keys:
                values = [m.get(key, 0.0) for m in self._all_metrics]
                # Convert tensor to float if needed
                values = [float(v) if isinstance(v, torch.Tensor) else v for v in values]
                avg_metrics[key] = np.mean(values)

            # Append average row to CSV
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'AVERAGE',
                    avg_metrics.get('dice', 0.0),
                    avg_metrics.get('iou', 0.0),
                    avg_metrics.get('precision', 0.0),
                    avg_metrics.get('recall', 0.0),
                    avg_metrics.get('specificity', 0.0),
                    avg_metrics.get('cldice', 0.0),
                    avg_metrics.get('connectivity', 0.0),
                    avg_metrics.get('density_error', 0.0),
                    avg_metrics.get('betti_0_error', 0.0),
                    avg_metrics.get('betti_1_error', 0.0),
                ])

            print(f"\n✅ Average metrics saved to: {self.metrics_file}")
