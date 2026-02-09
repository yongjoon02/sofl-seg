"""MedSegDiff flow backbones with VLM-FiLM conditioning."""

import math
from typing import Iterable

import torch
import torch.nn as nn

from src.archs.components.diffusion_unet_vlm_film import MedSegDiffUNetVLMFiLM
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


@ARCHS_REGISTRY.register(name="medsegdiff_flow_vlm_film")
class MedSegDiffFlowVLMFiLM(nn.Module):
    """MedSegDiff UNet used as a flow-matching backbone with VLM-FiLM (flow head only)."""

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
        use_gradient_checkpointing: bool = False,
        **_,
    ):
        super().__init__()
        self.time_scale = float(time_scale)

        full_self_attn = _attn_flags_from_resolutions(
            image_size=img_resolution,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
        )

        self.base_unet = MedSegDiffUNetVLMFiLM(
            dim=model_channels,
            image_size=img_resolution,
            mask_channels=1,
            input_img_channels=1,
            dim_mult=tuple(channel_mult),
            full_self_attn=full_self_attn,
            mid_transformer_depth=1,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def forward(
        self,
        x,
        time,
        cond,
        vlm_cond: dict | None = None,
        vlm_film_heads=None,
        junction_gate: torch.Tensor | None = None,
        junction_warmup_active: bool = False,
    ):
        if isinstance(time, torch.Tensor):
            time_in = time * self.time_scale
        else:
            time_in = time * self.time_scale
        return self.base_unet(
            x,
            time_in,
            cond,
            vlm_cond=vlm_cond,
            vlm_film_heads=vlm_film_heads,
            junction_gate=junction_gate,
            junction_warmup_active=junction_warmup_active,
        )


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


@ARCHS_REGISTRY.register(name="medsegdiff_flow_multitask_vlm_film")
class MedSegDiffFlowMultiTaskVLMFiLM(nn.Module):
    """MedSegDiff multi-task flow backbone with VLM-FiLM conditioning."""

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
        use_refine: bool = False,
        refine_depth: int = 2,
        use_gradient_checkpointing: bool = False,
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

        self.base_unet = MedSegDiffUNetVLMFiLM(
            dim=model_channels,
            image_size=img_resolution,
            mask_channels=2,
            input_img_channels=1,
            dim_mult=tuple(channel_mult),
            full_self_attn=full_self_attn,
            mid_transformer_depth=1,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self._feat = None
        self.base_unet.final_res_block.register_forward_hook(self._capture_features)

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

        self.hard_head = nn.Conv2d(model_channels, 1, 1)
        self.soft_head = nn.Conv2d(model_channels, 1, 1)

    def _capture_features(self, _module, _inputs, output):
        self._feat = output

    def forward(
        self,
        x,
        time,
        cond,
        vlm_cond: dict | None = None,
        vlm_film_heads=None,
        return_features: bool = False,
        junction_gate: torch.Tensor | None = None,
        junction_warmup_active: bool = False,
    ):
        if isinstance(time, torch.Tensor):
            time_in = time * self.time_scale
        else:
            time_in = time * self.time_scale

        # Note: Multitask model uses base_unet which handles FiLM internally
        # FiLM is applied at post-concat in Stage 4/3 only via MedSegDiffUNetVLMFiLM
        _ = self.base_unet(
            x,
            time_in,
            cond,
            vlm_cond=vlm_cond,
            vlm_film_heads=vlm_film_heads,
            junction_gate=junction_gate,
            junction_warmup_active=junction_warmup_active,
        )
        feat = self._feat
        hard_feat = self.hard_refine(feat)
        soft_feat = self.soft_refine(feat)
        v_hard = self.hard_head(hard_feat)
        v_soft = self.soft_head(soft_feat)
        v_pred = torch.cat([v_hard, v_soft], dim=1)
        if return_features:
            return v_pred, feat
        return v_pred
