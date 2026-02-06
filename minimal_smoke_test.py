#!/usr/bin/env python3
"""Minimal smoke tests for VLM-FiLM port (no VLM model load)."""

from __future__ import annotations

import contextlib
import io
import json
import os

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


def _load_module(module_name: str, rel_path: str):
    base = Path(__file__).resolve().parent
    mod_path = base / rel_path
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vlm_conditioner = _load_module(
    "vlm_conditioner",
    "src/archs/components/vlm_conditioner.py",
)
_vlm_film = _load_module(
    "vlm_film",
    "src/archs/components/vlm_film.py",
)

AdaptiveFiLMHead = _vlm_conditioner.AdaptiveFiLMHead
ImageToConditionVector = _vlm_conditioner.ImageToConditionVector
VLMConditioner = _vlm_conditioner.VLMConditioner
apply_vlm_film = _vlm_film.apply_vlm_film


class DummyModule(nn.Module):
    def __init__(self, training: bool = True) -> None:
        super().__init__()
        self.training = training

    def train(self, mode: bool = True):
        self.training = mode
        return self


class DummyVLMGen:
    def __init__(self) -> None:
        self.last_vision_feat_source = "dummy"
        self.last_vision_feat_shape = (2, 5)

    def get_vision_embed_dim(self) -> int:
        return 4


class ConstFiLMHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, cond_vec: torch.Tensor):
        bsz = cond_vec.size(0)
        gamma = torch.ones((bsz, self.channels, 1, 1), device=cond_vec.device, dtype=cond_vec.dtype)
        beta = torch.zeros_like(gamma)
        return gamma, beta


def _capture_stdout(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return result, buf.getvalue()


def _extract_film_dbg(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("[FILM-DBG] "):
            payload = line.split(" ", 1)[1]
            return json.loads(payload)
    raise AssertionError("[FILM-DBG] not found in output")


def _make_dbg(step: int, bsz: int, cond_vec: torch.Tensor, stage: str = "train", should_log: bool = True) -> dict:
    cond_norm = cond_vec.detach().float().norm(dim=1)
    return {
        "stage": stage,
        "step": step,
        "B": int(bsz),
        "split_key": "train",
        "cache_key_example": "train:dummy",
        "cond_source": "image_global",
        "cond_dim": int(cond_vec.size(1)),
        "cond_norm_mean": cond_norm.mean().item(),
        "cond_norm_std": cond_norm.std(unbiased=False).item() if cond_norm.numel() > 1 else 0.0,
        "cond_cos_mean": 1.0,
        "cond_cos_min": 1.0,
        "cond_cos_max": 1.0,
        "cache_total": 0,
        "cache_hit": 0,
        "cache_miss": 0,
        "hit_rate": 0.0,
        "mem_hit": 0,
        "disk_hit": 0,
        "mixed_keys": False,
        "should_log": bool(should_log),
    }


def _run_apply(x: torch.Tensor, vlm_cond: dict, heads: nn.ModuleList, module: DummyModule):
    def _call():
        return apply_vlm_film(x, vlm_cond, heads, stage_idx=0, module=module)

    (out, _payload), stdout = _capture_stdout(_call)
    payload = _extract_film_dbg(stdout)
    return out, payload, stdout


def main() -> None:
    torch.manual_seed(0)

    bsz = 4
    cond_dim = 16
    channels = 8

    cond_vec = torch.randn(bsz, cond_dim)
    x = torch.randn(bsz, channels, 4, 4)
    heads = nn.ModuleList([AdaptiveFiLMHead(cond_dim=cond_dim, channels=channels, hidden_dim=32)])
    module = DummyModule(training=True)

    # VLM_FILM_MODE=on
    os.environ["VLM_FILM_MODE"] = "on"
    dbg = _make_dbg(step=1, bsz=bsz, cond_vec=cond_vec, stage="train", should_log=True)
    vlm_cond = {"cond_vec": cond_vec, "_dbg": dbg}
    out_on, payload_on, stdout_on = _run_apply(x.clone(), vlm_cond, heads, module)
    assert payload_on["film_mode"] == "on"
    for key in ("mean_abs_diff", "max_abs_diff", "gamma_dev_std", "beta_norm_std"):
        assert key in payload_on

    # VLM_FILM_MODE=off (bypass)
    os.environ["VLM_FILM_MODE"] = "off"
    out_off, payload_off, stdout_off = _run_apply(x.clone(), vlm_cond, heads, module)
    assert payload_off["film_mode"] == "off"
    assert payload_off["film_bypass"] is True
    assert torch.allclose(out_off, x, atol=1e-6)

    # VLM_FILM_MODE=shuffle (permute)
    os.environ["VLM_FILM_MODE"] = "shuffle"
    out_shuf, payload_shuf, stdout_shuf = _run_apply(x.clone(), vlm_cond, heads, module)
    assert "[FILM-SHUFFLE]" in stdout_shuf
    assert payload_shuf["shuffle_perm_hash"] is not None
    assert len(payload_shuf["shuffle_perm_hash"]) == 8

    # Feature dim mismatch (ImageToConditionVector)
    itc = ImageToConditionVector(feat_dim=4, cond_dim=8, use_layernorm=True)
    try:
        itc(torch.zeros(2, 5))
        raise AssertionError("Expected RuntimeError for ImageToConditionVector mismatch")
    except RuntimeError as exc:
        expected = (
            "[VLM] image_to_cond feature dim mismatch: "
            "got=5 expected=4 feat_shape=(2, 5) expected_feat_dim=4"
        )
        assert str(exc) == expected

    # Feature dim mismatch (_check_image_feat_dim)
    cond = VLMConditioner(enabled=False, verbose=False)
    cond.image_to_cond = ImageToConditionVector(feat_dim=4, cond_dim=8, use_layernorm=True)
    cond.vlm_generator = DummyVLMGen()
    try:
        cond._check_image_feat_dim(torch.zeros(2, 5))
        raise AssertionError("Expected RuntimeError for _check_image_feat_dim mismatch")
    except RuntimeError as exc:
        expected = (
            "[VLM] image_to_cond feature dim mismatch (fixed projection). "
            "got=5 expected=4 feat_batch_shape=(2, 5) vision_embed_dim=4 "
            "last_vision_feat_source=dummy last_vision_feat_shape=(2, 5)"
        )
        assert str(exc) == expected

    # [FILM-WARN] gamma/beta constant warning
    os.environ["VLM_FILM_MODE"] = "on"
    const_heads = nn.ModuleList([ConstFiLMHead(channels=channels)])
    dbg_const = _make_dbg(step=2, bsz=bsz, cond_vec=cond_vec, stage="train", should_log=True)
    vlm_cond_const = {"cond_vec": cond_vec, "_dbg": dbg_const}

    def _run_warn():
        for _ in range(5):
            apply_vlm_film(x.clone(), vlm_cond_const, const_heads, stage_idx=0, module=module)

    _, stdout_warn = _capture_stdout(_run_warn)
    assert "[FILM-WARN]" in stdout_warn

    # [FILM-VAL] summary on val->train transition
    os.environ["VLM_FILM_MODE"] = "on"
    module.train(False)
    dbg_val = _make_dbg(step=3, bsz=bsz, cond_vec=cond_vec, stage="val", should_log=True)
    vlm_cond_val = {"cond_vec": cond_vec, "_dbg": dbg_val}

    def _run_val():
        apply_vlm_film(x.clone(), vlm_cond_val, const_heads, stage_idx=0, module=module)
        apply_vlm_film(x.clone(), vlm_cond_val, const_heads, stage_idx=0, module=module)
        module.train(True)
        apply_vlm_film(x.clone(), vlm_cond_val, const_heads, stage_idx=0, module=module)

    _, stdout_val = _capture_stdout(_run_val)
    assert "[FILM-VAL]" in stdout_val

    print("OK: minimal_smoke_test")


if __name__ == "__main__":
    main()
