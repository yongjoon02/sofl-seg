"""SegDiff backbone adapted for flow matching (single-head flow output)."""

import math
from typing import Iterable

import torch
import torch.nn as nn

from src.archs.components.diffusion_unet import SegDiffUNet
from src.registry.base import ARCHS_REGISTRY


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


@ARCHS_REGISTRY.register(name="segdiff_flow")
class SegDiffFlow(nn.Module):
    """SegDiff UNet used as a flow-matching backbone (flow head only)."""

    def __init__(
        self,
        img_resolution: int,
        model_channels: int = 64,
        channel_mult: Iterable[int] = (1, 2, 4, 8),
        channel_mult_emb: int = 4,  # unused (kept for signature parity)
        num_blocks: int = 3,  # unused (SegDiff uses fixed depth per stage)
        attn_resolutions: Iterable[int] = (32, 16, 8, 8),
        dropout: float = 0.0,  # unused
        label_dim: int = 0,  # unused
        augment_dim: int = 0,  # unused
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        resnet_block_groups: int = 8,
        rrdb_blocks: int = 3,
        time_scale: float = 1000.0,  # scale flow t (0-1) to diffusion-style range
        **_,
    ):
        super().__init__()
        self.time_scale = float(time_scale)

        full_self_attn = _attn_flags_from_resolutions(
            image_size=img_resolution,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
        )

        self.base_unet = SegDiffUNet(
            dim=model_channels,
            image_size=img_resolution,
            mask_channels=1,
            input_img_channels=1,
            dim_mult=tuple(channel_mult),
            full_self_attn=full_self_attn,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
            resnet_block_groups=resnet_block_groups,
            rrdb_blocks=rrdb_blocks,
        )

    def forward(self, x, time, cond):
        if isinstance(time, torch.Tensor):
            time_in = time * self.time_scale
        else:
            time_in = time * self.time_scale
        return self.base_unet(x, time_in, cond)
