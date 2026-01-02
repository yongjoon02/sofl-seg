"""Composite loss for flow matching models (L1 + optional BCE/L2/Dice)."""

import torch
import torch.nn as nn

from src.registry.losses import register_loss


@register_loss(
    name='flow_matching',
    description='Weighted combination of flow-matching losses (L1/BCE/L2/Dice)',
    supports_multiclass=False,
    supports_soft_labels=True,
)
class FlowMatchingLoss(nn.Module):
    """
    Args:
        scheme: String like 'l1_bce_l2' to select components
        weights: Dict overriding weights per component (e.g., {'l1': 1.0, 'bce': 0.5})
        use_bce/use_l2/use_dice & *_weight: legacy flags kept for backward compatibility
    """

    VALID_COMPONENTS = {'l1', 'bce', 'l2', 'dice', 'l1geo', 'l1geo_head', 'bce_hard'}

    def __init__(
        self,
        scheme: str | None = None,
        weights: dict | None = None,
        t_threshold: float = 0.7,
        use_bce: bool | None = None,
        use_l2: bool | None = None,
        use_dice: bool | None = None,
        bce_weight: float = 0.5,
        hard_bce_weight: float = 0.1,
        l2_weight: float = 0.1,
        dice_weight: float = 0.2,
    ):
        super().__init__()
        self.t_threshold = float(t_threshold)

        # Determine active components
        if scheme:
            parts = [part for part in scheme.split('_') if part]
            components = []
            i = 0
            while i < len(parts):
                part = parts[i]
                if part == 'l1geo' and i + 1 < len(parts) and parts[i + 1] == 'head':
                    components.append('l1geo_head')
                    i += 2
                    continue
                if part == 'bce' and i + 1 < len(parts) and parts[i + 1] == 'hard':
                    components.append('bce_hard')
                    i += 2
                    continue
                components.append(part)
                i += 1
        else:
            components = []
            if use_bce:
                components.append('bce')
            if use_l2:
                components.append('l2')
            if use_dice:
                components.append('dice')
            # Default base term (when nothing specified): use l2 to match
            # the standard flow-matching objective.
            if not components:
                components = ['l2']

        # If scheme is explicitly provided, honor it as-is, but require a base
        # matching term (l1 or l2) to avoid training with only regularizers.
        if scheme and ('l1' not in components and 'l2' not in components):
            raise ValueError(
                f"Invalid flow_matching scheme '{scheme}': must include 'l1' or 'l2' base term."
            )

        # Remove duplicates while preserving order
        seen = set()
        ordered_components = []
        for comp in components:
            if comp in self.VALID_COMPONENTS and comp not in seen:
                seen.add(comp)
                ordered_components.append(comp)
        if not ordered_components:
            raise ValueError(f"No valid loss components parsed from scheme='{scheme}'")
        self.components = ordered_components

        # Setup weights (defaults + overrides)
        self.weights = {comp: 1.0 for comp in self.components}
        legacy_weights = {
            'bce': bce_weight,
            'bce_hard': hard_bce_weight,
            'l2': l2_weight,
            'dice': dice_weight,
        }
        for comp, value in legacy_weights.items():
            if comp in self.weights:
                self.weights[comp] = value
        if weights:
            for comp, value in weights.items():
                if comp in self.VALID_COMPONENTS:
                    self.weights[comp] = value

        # Initialize Dice loss lazily
        if 'dice' in self.components:
            from src.losses.dice_loss import DiceLoss
            self.dice_loss = DiceLoss()

    def forward(self, v, ut, xt, geometry, t=None, geometry_pred=None, hard_labels=None):
        """
        Args:
            v: predicted flow
            ut: target flow
            xt: current state (proxy for geometry prob map)
            geometry: target geometry map
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total = 0.0

        if 'l1' in self.components:
            l1_loss = torch.abs(v - ut).mean()
            losses['l1'] = l1_loss
            total = total + self.weights.get('l1', 1.0) * l1_loss

        if 'bce' in self.components:
            output_geometry = xt
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry

            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)

            eps = 1e-7
            output_probs = torch.clamp(output_probs, eps, 1 - eps)
            bce_loss = -(target_probs * torch.log(output_probs) +
                        (1 - target_probs) * torch.log(1 - output_probs)).mean()
            losses['bce'] = bce_loss
            total = total + self.weights.get('bce', 1.0) * bce_loss

        if 'bce_hard' in self.components:
            if geometry_pred is None or hard_labels is None:
                raise ValueError("bce_hard requires geometry_pred and hard_labels inputs.")

            pred = geometry_pred
            if pred.dim() == 4 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if hard_labels.dim() == 4 and hard_labels.shape[1] == 1:
                hard_2d = hard_labels.squeeze(1)
            else:
                hard_2d = hard_labels

            pred_probs = torch.clamp(pred, 0.0, 1.0)
            target_probs = torch.clamp(hard_2d, 0.0, 1.0)

            eps = 1e-7
            pred_probs = torch.clamp(pred_probs, eps, 1 - eps)
            bce_hard = -(target_probs * torch.log(pred_probs) +
                        (1 - target_probs) * torch.log(1 - pred_probs)).mean()
            losses['bce_hard'] = bce_hard
            total = total + self.weights.get('bce_hard', 1.0) * bce_hard

        if 'l2' in self.components:
            l2_loss = ((v - ut) ** 2).mean()
            losses['l2'] = l2_loss
            total = total + self.weights.get('l2', 1.0) * l2_loss

        if 'l1geo' in self.components:
            output_geometry = xt
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry

            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            l1_geo_loss = torch.abs(output_probs - target_probs).mean()
            losses['l1geo'] = l1_geo_loss
            total = total + self.weights.get('l1geo', 1.0) * l1_geo_loss

        if 'l1geo_head' in self.components:
            if geometry_pred is None or t is None:
                raise ValueError("l1geo_head requires geometry_pred and t inputs.")

            pred = geometry_pred
            if pred.dim() == 4 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry

            pred_probs = torch.clamp(pred, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)

            per_sample = torch.abs(pred_probs - target_probs).mean(dim=(1, 2))
            gate = (t > self.t_threshold).float()
            l1_geo_head = (per_sample * gate).sum() / (gate.sum() + 1e-8)
            losses['l1geo_head'] = l1_geo_head
            total = total + self.weights.get('l1geo_head', 1.0) * l1_geo_head

        if 'dice' in self.components:
            output_geometry = xt
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry

            output_probs = torch.clamp(output_geometry, 0.0, 1.0).unsqueeze(1)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            dice_loss = self.dice_loss(output_probs, target_probs)
            losses['dice'] = dice_loss
            total = total + self.weights.get('dice', 1.0) * dice_loss

        return total, losses
