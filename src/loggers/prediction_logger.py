"""Custom logger for saving predictions and metrics."""

import csv
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize
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
        """Save predictions, labels, and per-sample metrics."""
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

            # Compute per-sample metrics before transpose/save.
            sample_metrics = self._compute_sample_metrics(pred, label)

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

            # Save metrics to CSV (use per-sample metrics, not aggregated batch metrics)
            self._save_sample_metrics(filename, sample_metrics)

    @staticmethod
    def _compute_sample_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
        """Compute per-sample binary segmentation metrics."""
        pred_bin = pred > 0.5
        label_bin = label > 0.5

        tp = np.logical_and(pred_bin, label_bin).sum()
        fp = np.logical_and(pred_bin, ~label_bin).sum()
        fn = np.logical_and(~pred_bin, label_bin).sum()
        tn = np.logical_and(~pred_bin, ~label_bin).sum()

        eps = 1e-8
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)

        # clDice (centerline Dice)
        pred_skel = skeletonize(pred_bin).astype(np.float32)
        label_skel = skeletonize(label_bin).astype(np.float32)
        tprec = (pred_skel * label_bin).sum() / (pred_skel.sum() + 1e-5)
        tsens = (pred_bin * label_skel).sum() / (label_skel.sum() + 1e-5)
        cldice = 2 * tprec * tsens / (tprec + tsens + 1e-5)

        # Betti errors
        pred_b0 = ndimage.label(pred_bin)[1]
        label_b0 = ndimage.label(label_bin)[1]
        betti_0_error = abs(pred_b0 - label_b0)

        pred_eroded = ndimage.binary_erosion(pred_bin)
        label_eroded = ndimage.binary_erosion(label_bin)
        pred_b1 = max(0, ndimage.label(pred_eroded)[1] - pred_b0)
        label_b1 = max(0, ndimage.label(label_eroded)[1] - label_b0)
        betti_1_error = abs(pred_b1 - label_b1)

        return {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'cldice': float(cldice),
            'connectivity': 0.0,
            'density_error': 0.0,
            'betti_0_error': float(betti_0_error),
            'betti_1_error': float(betti_1_error),
        }

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
            metric_keys = ['dice', 'iou', 'precision', 'recall', 'specificity',
                          'cldice', 'connectivity', 'density_error',
                          'betti_0_error', 'betti_1_error']

            stats = {}
            for key in metric_keys:
                values = [float(m.get(key, 0.0)) for m in self._all_metrics]
                stats[key] = (np.mean(values), np.std(values))

            # Append mean±std row to CSV
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'AVERAGE±STD',
                    f"{stats['dice'][0]:.6f}±{stats['dice'][1]:.6f}",
                    f"{stats['iou'][0]:.6f}±{stats['iou'][1]:.6f}",
                    f"{stats['precision'][0]:.6f}±{stats['precision'][1]:.6f}",
                    f"{stats['recall'][0]:.6f}±{stats['recall'][1]:.6f}",
                    f"{stats['specificity'][0]:.6f}±{stats['specificity'][1]:.6f}",
                    f"{stats['cldice'][0]:.6f}±{stats['cldice'][1]:.6f}",
                    f"{stats['connectivity'][0]:.6f}±{stats['connectivity'][1]:.6f}",
                    f"{stats['density_error'][0]:.6f}±{stats['density_error'][1]:.6f}",
                    f"{stats['betti_0_error'][0]:.6f}±{stats['betti_0_error'][1]:.6f}",
                    f"{stats['betti_1_error'][0]:.6f}±{stats['betti_1_error'][1]:.6f}",
                ])

            print(f"\n✅ Average metrics (mean±std) saved to: {self.metrics_file}")
