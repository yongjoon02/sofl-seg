"""MedSegDiff flow backbone with 2-channel output (multi-task with late-decoder split + refinement)."""
import math
from typing import Iterable

import torch
import torch.nn as nn

from src.archs.components.diffusion_unet import MedSegDiffUNet
from src.registry.base import ARCHS_REGISTRY


class TaskSpecificRefinementBlock(nn.Module):
    """Task-specific refinement block: Conv + GroupNorm + SiLU, repeated."""

    def __init__(self, channels: int, num_layers: int = 2, num_groups: int = 8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, channels),
                nn.SiLU(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _attn_flags_from_resolutions(
    image_size: int,
    channel_mult: Iterable[int],
    attn_resolutions: Iterable[int],
) -> tuple[bool, ...]:
    """Convert resolution list to per-level attention flags."""
    attn_set = set(attn_resolutions)
    flags = []
    curr = image_size
    for _ in channel_mult:
        flags.append(curr in attn_set)
        curr = max(1, math.floor(curr / 2))
    return tuple(flags)


@ARCHS_REGISTRY.register(name="medsegdiff_flow_multitask")
class MedSegDiffFlowMultiTask(nn.Module):
    """MedSegDiff UNet for multi-task flow matching (hard + soft).

    Two modes controlled by use_refine:
        1. Head-only (use_refine=false): feat -> 1x1 conv head
        2. Refine-on (use_refine=true): feat -> refine blocks -> 1x1 conv head

    Forward output: v_pred (velocity) with 2 channels
        - Channel 0: hard
        - Channel 1: soft
    """

    def __init__(
        self,
        img_resolution: int,
        model_channels: int = 64,
        channel_mult: Iterable[int] = (1, 2, 4, 8),
        channel_mult_emb: int = 4,  # unused
        num_blocks: int = 3,  # unused
        attn_resolutions: Iterable[int] = (32, 16, 8, 8),
        dropout: float = 0.0,  # unused
        label_dim: int = 0,  # unused
        augment_dim: int = 0,  # unused
        time_scale: float = 1000.0,
        use_refine: bool = False,  # enable task-specific refinement blocks
        refine_depth: int = 2,  # number of Conv+GN+SiLU layers
        **_,
    ):
        super().__init__()
        self.time_scale = float(time_scale)
        self.model_channels = model_channels
        self.use_refine = use_refine
        self.refine_depth = refine_depth

        full_self_attn = _attn_flags_from_resolutions(
            image_size=img_resolution,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
        )

        # Shared encoder-decoder
        self.base_unet = MedSegDiffUNet(
            dim=model_channels,
            image_size=img_resolution,
            mask_channels=2,
            input_img_channels=1,
            dim_mult=tuple(channel_mult),
            full_self_attn=full_self_attn,
            mid_transformer_depth=1,
        )
        self._feat = None
        self.base_unet.final_res_block.register_forward_hook(self._capture_features)

        # Task-specific modules (conditional on use_refine)
        if use_refine:
            self.hard_refine = TaskSpecificRefinementBlock(
                model_channels,
                num_layers=refine_depth,
            )
            self.soft_refine = TaskSpecificRefinementBlock(
                model_channels,
                num_layers=refine_depth,
            )
        else:
            self.hard_refine = nn.Identity()
            self.soft_refine = nn.Identity()

        # Logit heads (1x1 conv): predict velocity v
        self.hard_head = nn.Conv2d(model_channels, 1, 1)
        self.soft_head = nn.Conv2d(model_channels, 1, 1)

    def _capture_features(self, _module, _inputs, output):
        self._feat = output

    def forward(self, x, time, cond, return_features=False):
        """Forward pass predicting velocity v.

        Args:
            x: Noisy input (B, 2, H, W)
            time: Flow time t in [0, 1] (B,)
            cond: Conditional image (B, 1, H, W)
            return_features: If True, return intermediate features

        Returns:
            v_pred (B, 2, H, W): Velocity field, Channel 0=hard, 1=soft
        """
        if isinstance(time, torch.Tensor):
            time_in = time * self.time_scale
        else:
            time_in = time * self.time_scale

        _ = self.base_unet(x, time_in, cond)
        feat = self._feat
        if feat is None:
            raise RuntimeError("Failed to capture features from MedSegDiffUNet.")

        hard_feat = self.hard_refine(feat)
        soft_feat = self.soft_refine(feat)

        v_hard = self.hard_head(hard_feat)
        v_soft = self.soft_head(soft_feat)

        v_pred = torch.cat([v_hard, v_soft], dim=1)

        if return_features:
            return v_pred, feat, hard_feat, soft_feat
        return v_pred
