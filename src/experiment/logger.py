"""Enhanced TensorBoard logger with rich logging capabilities."""

from pathlib import Path

import torch
import torchvision
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only


class EnhancedTensorBoardLogger(TensorBoardLogger):
    """
    Enhanced TensorBoard logger with additional features:
    - Automatic image logging
    - Hyperparameter tracking
    - Model graph visualization
    - Prediction visualization
    """

    def __init__(self, save_dir, name="", version=None, **kwargs):
        # Ensure save_dir exists before initializing parent
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        super().__init__(save_dir=save_dir, name=name, version=version, **kwargs)

        # Ensure log_dir exists
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        self.predictions_dir = Path(self.log_dir) / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def log_hyperparams_with_metrics(self, params, metrics):
        """Log hyperparameters along with initial metrics."""
        # Convert params to dict if needed
        if hasattr(params, '__dict__'):
            params = vars(params)

        # Log to tensorboard
        self.experiment.add_hparams(
            hparam_dict=params,
            metric_dict=metrics,
        )

    @rank_zero_only
    def log_images(self, tag, images, step=None):
        """
        Log images to TensorBoard.
        
        Args:
            tag: Name for the image group
            images: Tensor of shape (B, C, H, W) or list of tensors
            step: Global step
        """
        if isinstance(images, list):
            images = torch.stack(images)

        # Normalize to [0, 1] if needed
        if images.min() < 0 or images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)

        # Create grid
        grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)

        self.experiment.add_image(tag, grid, global_step=step)

    @rank_zero_only
    def log_prediction_comparison(self, images, predictions, labels, step, max_samples=4):
        """
        Log side-by-side comparison of input, prediction, and ground truth.
        
        Args:
            images: Input images (B, C, H, W)
            predictions: Model predictions (B, C, H, W) or (B, H, W)
            labels: Ground truth (B, C, H, W) or (B, H, W)
            step: Global step
            max_samples: Maximum number of samples to log
        """
        # Take subset
        images = images[:max_samples]
        predictions = predictions[:max_samples]
        labels = labels[:max_samples]

        # Ensure 3D tensors (B, H, W)
        if predictions.dim() == 4:
            predictions = predictions.squeeze(1)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        if images.dim() == 4 and images.shape[1] == 1:
            images = images.squeeze(1)

        # Normalize images to [0, 1]
        if images.min() < 0 or images.max() > 1:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)

        # Convert predictions and labels to [0, 1]
        predictions = (predictions > 0.5).float() if predictions.dtype == torch.float else predictions.float()
        labels = (labels > 0.5).float() if labels.dtype == torch.float else labels.float()

        # Stack: [image, prediction, label] for each sample
        # Add channel dimension for grayscale
        images = images.unsqueeze(1) if images.dim() == 3 else images
        predictions = predictions.unsqueeze(1)
        labels = labels.unsqueeze(1)

        # Interleave: img, pred, label, img, pred, label, ...
        comparison = torch.stack([images, predictions, labels], dim=1)  # (B, 3, C, H, W)
        comparison = comparison.reshape(-1, *comparison.shape[2:])  # (B*3, C, H, W)

        # Create grid
        grid = torchvision.utils.make_grid(comparison, nrow=3, normalize=False, pad_value=1.0)

        self.experiment.add_image('predictions/comparison', grid, global_step=step)

    @rank_zero_only
    def log_model_graph(self, model, input_shape=(1, 1, 224, 224)):
        """Log model computational graph."""
        try:
            dummy_input = torch.zeros(input_shape)
            self.experiment.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    @rank_zero_only
    def log_confusion_matrix(self, cm, class_names, step):
        """Log confusion matrix as image."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        # Convert to tensor
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = torch.from_numpy(image).permute(2, 0, 1)

        self.experiment.add_image('metrics/confusion_matrix', image, global_step=step)
        plt.close(fig)
