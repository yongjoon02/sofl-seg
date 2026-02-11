"""FiLM application helpers for VLM-FiLM conditioning (ported from soft-seg)."""

from __future__ import annotations

import hashlib
import json
import os

import torch
import torch.nn as nn
import torch.distributed as dist

_FILM_VAL_ACCUM = {
    "count": 0,
    "cond_norm_mean_sum": 0.0,
    "cond_norm_std_sum": 0.0,
    "gamma_dev_mean_sum": 0.0,
    "beta_norm_mean_sum": 0.0,
    "x_rel_delta_mean_sum": 0.0,
}
_FILM_LAST_MODE = None
_FILM_CONST_COUNT = 0
_FILM_SHUFFLE_LOGGED = False
# NEW: Track FiLM application counts per stage (for validation logging)
_FILM_APPLY_COUNT_PER_STAGE = {0: 0, 1: 0, 2: 0, 3: 0}
_FILM_APPLY_COUNT_LAST_STEP = -1


def _film_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _film_bypass_enabled() -> bool:
    return _film_mode() == "off"


def _film_mode() -> str:
    mode = os.getenv("VLM_FILM_MODE", "on").strip().lower()
    if mode not in ("on", "off", "shuffle"):
        mode = "on"
    return mode


def _film_shuffle_perm(batch: int, step: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(step) + 1234)
    return torch.randperm(batch, generator=gen, device=device)


def _film_update_val_accum(module: nn.Module, payload: dict) -> None:
    global _FILM_LAST_MODE, _FILM_SHUFFLE_LOGGED
    if not _film_rank0():
        return
    mode = "train" if module.training else "val"
    if _FILM_LAST_MODE == "val" and mode == "train":
        if _FILM_VAL_ACCUM["count"] > 0:
            cnt = _FILM_VAL_ACCUM["count"]
            summary = {
                "epoch": None,
                "val_dice": None,
                "val_iou": None,
                "pred_pos_ratio": None,
                "gt_pos_ratio": None,
                "cond_norm_mean_epoch": _FILM_VAL_ACCUM["cond_norm_mean_sum"] / cnt,
                "cond_norm_std_epoch": _FILM_VAL_ACCUM["cond_norm_std_sum"] / cnt,
                "gamma_dev_mean_epoch": _FILM_VAL_ACCUM["gamma_dev_mean_sum"] / cnt,
                "beta_norm_mean_epoch": _FILM_VAL_ACCUM["beta_norm_mean_sum"] / cnt,
                "x_rel_delta_mean_epoch": _FILM_VAL_ACCUM["x_rel_delta_mean_sum"] / cnt,
            }
            print(f"[FILM-VAL] {json.dumps(summary, ensure_ascii=True)}")
        _FILM_VAL_ACCUM["count"] = 0
        _FILM_VAL_ACCUM["cond_norm_mean_sum"] = 0.0
        _FILM_VAL_ACCUM["cond_norm_std_sum"] = 0.0
        _FILM_VAL_ACCUM["gamma_dev_mean_sum"] = 0.0
        _FILM_VAL_ACCUM["beta_norm_mean_sum"] = 0.0
        _FILM_VAL_ACCUM["x_rel_delta_mean_sum"] = 0.0
        _FILM_SHUFFLE_LOGGED = False

    if mode == "val":
        _FILM_VAL_ACCUM["count"] += 1
        _FILM_VAL_ACCUM["cond_norm_mean_sum"] += float(payload.get("cond_norm_mean", 0.0) or 0.0)
        _FILM_VAL_ACCUM["cond_norm_std_sum"] += float(payload.get("cond_norm_std", 0.0) or 0.0)
        _FILM_VAL_ACCUM["gamma_dev_mean_sum"] += float(payload.get("gamma_dev_mean", 0.0) or 0.0)
        _FILM_VAL_ACCUM["beta_norm_mean_sum"] += float(payload.get("beta_norm_mean", 0.0) or 0.0)
        _FILM_VAL_ACCUM["x_rel_delta_mean_sum"] += float(payload.get("rel_delta_mean", 0.0) or 0.0)

    _FILM_LAST_MODE = mode


def apply_vlm_film(
    x: torch.Tensor,
    vlm_cond: dict | None,
    vlm_film_heads: nn.ModuleList | None,
    stage_idx: int,
    module: nn.Module,
    apply_point: str = "legacy_post_block2",
    junction_gate: torch.Tensor | None = None,
    junction_warmup_active: bool = False,
) -> tuple[torch.Tensor, dict | None]:
    """Apply VLM-FiLM to a feature map, returning (x_post, payload_or_none).
    
    Args:
        x: Feature tensor [B, C, H, W]
        vlm_cond: VLM conditioning dict with 'cond_vec'
        vlm_film_heads: ModuleList of FiLM heads (only Stage 4/3 heads exist)
        stage_idx: Decoder stage index (0=Layer4, 1=Layer3, 2=Layer2, 3=Layer1)
        module: Parent module for training flag
        apply_point: Application point label for logging
        junction_gate: Optional spatial gate [B, 1, H, W], where gate=1 enables FiLM,
                      gate=0 disables FiLM at those spatial locations (junction safety)
        junction_warmup_active: Whether junction gating is in warm-up period (gates=1.0)
    """
    global _FILM_APPLY_COUNT_PER_STAGE, _FILM_APPLY_COUNT_LAST_STEP
    
    if vlm_cond is None or vlm_film_heads is None:
        return x, None
    if stage_idx >= len(vlm_film_heads):
        return x, None
    
    # Track application counts per stage
    dbg = vlm_cond.get("_dbg") if isinstance(vlm_cond, dict) else None
    current_step = int(dbg.get("step", 0)) if dbg else 0
    if current_step != _FILM_APPLY_COUNT_LAST_STEP:
        # Reset counts for new step
        _FILM_APPLY_COUNT_PER_STAGE = {0: 0, 1: 0, 2: 0, 3: 0}
        _FILM_APPLY_COUNT_LAST_STEP = current_step
    _FILM_APPLY_COUNT_PER_STAGE[stage_idx] += 1

    film_mode = _film_mode()
    cond_vec = vlm_cond["cond_vec"]
    perm = None
    shuffle_perm_hash = None
    if film_mode == "shuffle":
        perm = _film_shuffle_perm(
            cond_vec.size(0),
            int(vlm_cond.get("_dbg", {}).get("step", 0)),
            cond_vec.device,
        )
        cond_vec = cond_vec.index_select(0, perm)
        if _film_rank0():
            perm_list = perm.detach().cpu().tolist()
            shuffle_perm_hash = hashlib.md5(str(perm_list).encode()).hexdigest()[:8]

    gamma, beta = vlm_film_heads[stage_idx](cond_vec)
    if gamma.dtype != x.dtype:
        gamma = gamma.to(dtype=x.dtype)
    if beta.dtype != x.dtype:
        beta = beta.to(dtype=x.dtype)
    x_pre = x
    if film_mode == "off":
        x_post = x_pre
    else:
        # Apply FiLM modulation
        x_modulated = gamma * x_pre + beta
        
        # Apply spatial gating if provided (junction-aware FiLM gating)
        if junction_gate is not None:
            # gate shape: (B, 1, H, W), broadcast across channels
            # x_post = x_pre + gate * (x_modulated - x_pre)
            # gate=1: full FiLM, gate=0: no FiLM (identity)
            if junction_gate.shape[2:] != x_pre.shape[2:]:
                # Resize gate to match feature resolution
                import torch.nn.functional as F
                junction_gate = F.interpolate(
                    junction_gate,
                    size=x_pre.shape[2:],
                    mode='nearest'
                )
            x_post = x_pre + junction_gate * (x_modulated - x_pre)
        else:
            x_post = x_modulated
    x = x_post

    dbg = vlm_cond.get("_dbg") if isinstance(vlm_cond, dict) else None
    payload = None
    if dbg and stage_idx == 0 and _film_rank0():
        do_stats = (not module.training) or bool(dbg.get("should_log"))
        if do_stats:
            with torch.no_grad():
                gamma_det = gamma.detach().float()
                beta_det = beta.detach().float()
                x_pre_det = x_pre.detach().float()
                x_post_det = x_post.detach().float()

                gamma_mean = gamma_det.mean().item()
                gamma_std = gamma_det.std(unbiased=False).item()
                gamma_dev_mean = (gamma_det - 1.0).abs().mean().item()
                beta_mean = beta_det.mean().item()
                beta_std = beta_det.std(unbiased=False).item()
                beta_norm_mean = beta_det.view(beta_det.size(0), -1).norm(dim=1).mean().item()

                x_pre_flat = x_pre_det.view(x_pre_det.size(0), -1)
                x_post_flat = x_post_det.view(x_post_det.size(0), -1)
                x_pre_norm = x_pre_flat.norm(dim=1)
                x_post_norm = x_post_flat.norm(dim=1)
                x_pre_norm_mean = x_pre_norm.mean().item()
                x_post_norm_mean = x_post_norm.mean().item()
                diff = (x_post_det - x_pre_det).abs()
                mean_abs_diff = diff.mean().item()
                max_abs_diff = diff.max().item()
                rel_per_sample = (x_post_flat - x_pre_flat).norm(dim=1) / (x_pre_norm + 1e-6)
                rel_delta_mean = rel_per_sample.mean().item()
                rel_delta_p50 = torch.quantile(rel_per_sample, 0.5).item()
                rel_delta_p95 = torch.quantile(rel_per_sample, 0.95).item()
                
                # Junction gate statistics
                gate_mean = None
                gate_min = None
                gate_max = None
                gate_junction_pct = None
                if junction_gate is not None:
                    gate_det = junction_gate.detach().float()
                    gate_mean = gate_det.mean().item()
                    gate_min = gate_det.min().item()
                    gate_max = gate_det.max().item()
                    # Percentage of pixels where gate < 0.99 (junction regions)
                    gate_junction_pct = ((gate_det < 0.99).float().mean() * 100).item()

                payload = {
                    "stage": dbg.get("stage"),
                    "step": dbg.get("step"),
                    "B": dbg.get("B"),
                    "split_key": dbg.get("split_key"),
                    "cache_key_example": dbg.get("cache_key_example"),
                    "cond_source": dbg.get("cond_source"),
                    "cond_dim": dbg.get("cond_dim"),
                    "cond_norm_mean": dbg.get("cond_norm_mean"),
                    "cond_norm_std": dbg.get("cond_norm_std"),
                    "cond_cos_mean": dbg.get("cond_cos_mean"),
                    "cond_cos_min": dbg.get("cond_cos_min"),
                    "cond_cos_max": dbg.get("cond_cos_max"),
                    "vlm_response_debug": dbg.get("vlm_response_debug"),
                    "film_mode": film_mode,
                    "gamma_mean": gamma_mean,
                    "gamma_std": gamma_std,
                    "gamma_dev_mean": gamma_dev_mean,
                    "gamma_dev_std": (gamma_det - 1.0).std(unbiased=False).item(),
                    "beta_mean": beta_mean,
                    "beta_std": beta_std,
                    "beta_norm_mean": beta_norm_mean,
                    "beta_norm_std": beta_det.view(beta_det.size(0), -1).norm(dim=1).std(unbiased=False).item(),
                    "x_pre_norm_mean": x_pre_norm_mean,
                    "x_post_norm_mean": x_post_norm_mean,
                    "mean_abs_diff": mean_abs_diff,
                    "max_abs_diff": max_abs_diff,
                    "rel_delta_mean": rel_delta_mean,
                    "rel_delta_p50": rel_delta_p50,
                    "rel_delta_p95": rel_delta_p95,
                    "cache_total": dbg.get("cache_total"),
                    "cache_hit": dbg.get("cache_hit"),
                    "cache_miss": dbg.get("cache_miss"),
                    "hit_rate": dbg.get("hit_rate"),
                    "mem_hit": dbg.get("mem_hit"),
                    "disk_hit": dbg.get("disk_hit"),
                    "mixed_keys": dbg.get("mixed_keys"),
                    "film_bypass": _film_bypass_enabled(),
                    "shuffle_perm_hash": shuffle_perm_hash,
                    # NEW: Enhanced logging for Stage 4/3 post-concat strategy
                    "film_apply_point": apply_point,
                    "film_stage_idx": stage_idx,
                    "film_stage_layer": 4 - stage_idx,  # stage_idx=0→Layer4, =1→Layer3
                    "x_shape_at_apply": list(x_pre.shape),
                    "legacy_hook_path_disabled": True,
                    "film_apply_stage_ids": [0, 1],  # Only Stage 4/3
                    "film_apply_count_per_stage": dict(_FILM_APPLY_COUNT_PER_STAGE),
                    # Junction gating stats
                    "junction_gate_enabled": junction_gate is not None,
                    "junction_gate_warmup_active": junction_warmup_active,
                    "junction_gate_mean": gate_mean,
                    "junction_gate_min": gate_min,
                    "junction_gate_max": gate_max,
                    "junction_area_pct": gate_junction_pct,
                }
                if dbg.get("should_log"):
                    print(f"[FILM-DBG] {json.dumps(payload, ensure_ascii=True)}")
                _film_update_val_accum(module, payload)
                global _FILM_CONST_COUNT, _FILM_SHUFFLE_LOGGED
                if gamma_std < 1e-6 and beta_std < 1e-6:
                    _FILM_CONST_COUNT += 1
                    if _FILM_CONST_COUNT >= 5:
                        print("[FILM-WARN] gamma/beta appear constant for >=5 intervals")
                        _FILM_CONST_COUNT = 0
                else:
                    _FILM_CONST_COUNT = 0
                if film_mode == "off":
                    assert max_abs_diff < 1e-6, "FiLM off should be identity"
                if film_mode == "shuffle" and not _FILM_SHUFFLE_LOGGED and perm is not None:
                    print(f"[FILM-SHUFFLE] perm_first10={perm[:10].detach().cpu().tolist()}")
                    _FILM_SHUFFLE_LOGGED = True

    return x, payload
