"""Vessel-specific segmentation metrics based on torchmetrics."""

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize
from torchmetrics import Metric


class clDice(Metric):
    """Centerline Dice for vessel segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        self.add_state("cldice_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _get_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Get skeleton of binary mask."""
        return skeletonize(mask > 0.5).astype(np.float32)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (B, C, H, W) or (B, H, W) prediction
            target: (B, H, W) target
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        # Convert to numpy for skeletonization
        pred_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()

        batch_size = pred_np.shape[0]

        for i in range(batch_size):
            pred_skel = self._get_skeleton(pred_np[i])
            target_skel = self._get_skeleton(target_np[i])

            # clDice = 2 * tprec * tsens / (tprec + tsens)
            pred_skel_torch = torch.from_numpy(pred_skel).to(preds.device)
            target_torch = target[i].float()

            tprec = (pred_skel_torch * target_torch).sum() / (pred_skel_torch.sum() + self.smooth)

            target_skel_torch = torch.from_numpy(target_skel).to(preds.device)
            pred_torch = preds[i].float()

            tsens = (pred_torch * target_skel_torch).sum() / (target_skel_torch.sum() + self.smooth)

            cldice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
            self.cldice_sum += cldice
            self.total += 1

    def compute(self):
        return self.cldice_sum / self.total


class Betti0Error(Metric):
    """Betti-0 error (connected components error)."""

    def __init__(self):
        super().__init__()
        self.add_state("betti_0_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_betti_0(self, mask: np.ndarray) -> int:
        """Compute Betti-0 (number of connected components)."""
        labeled, num_components = ndimage.label(mask > 0.5)
        return num_components

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        pred_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()

        batch_size = pred_np.shape[0]

        for i in range(batch_size):
            pred_b0 = self._compute_betti_0(pred_np[i])
            target_b0 = self._compute_betti_0(target_np[i])
            self.betti_0_error += abs(pred_b0 - target_b0)
            self.total += 1

    def compute(self):
        return self.betti_0_error / self.total


class Betti1Error(Metric):
    """Betti-1 error (holes error)."""

    def __init__(self):
        super().__init__()
        self.add_state("betti_1_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_betti_1(self, mask: np.ndarray) -> int:
        """Compute Betti-1 (number of holes) - approximation."""
        labeled, num_components = ndimage.label(mask > 0.5)
        eroded = ndimage.binary_erosion(mask > 0.5)
        labeled_eroded, num_eroded = ndimage.label(eroded)
        betti_1 = max(0, num_eroded - num_components)
        return betti_1

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        pred_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()

        batch_size = pred_np.shape[0]

        for i in range(batch_size):
            pred_b1 = self._compute_betti_1(pred_np[i])
            target_b1 = self._compute_betti_1(target_np[i])
            self.betti_1_error += abs(pred_b1 - target_b1)
            self.total += 1

    def compute(self):
        return self.betti_1_error / self.total

