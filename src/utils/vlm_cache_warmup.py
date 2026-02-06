"""VLM cache warmup utility (precompute per-image VLM outputs)."""

from __future__ import annotations

import inspect
import time
from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.registry import get_dataset_info
from src.archs.components.vlm_conditioner import VLMConditioner


def _filter_kwargs(cls, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(cls.__init__)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs


def _build_dataset(
    dataset_cls,
    split_dir: str,
    data_cfg: dict,
    dataset_info,
    use_sliding_window: Optional[bool] = None,
):
    kwargs = {
        "path": split_dir,
        "augmentation": False,
        "crop_size": data_cfg.get("crop_size", dataset_info.default_crop_size),
        "num_samples_per_image": 1,
        "use_sliding_window": data_cfg.get("use_sliding_window", False) if use_sliding_window is None else use_sliding_window,
        "sliding_window_overlap": data_cfg.get("sliding_window_overlap", 0.25),
    }
    # Optional dataset-specific params
    for key in ("label_subdir", "use_sauna_transform"):
        if key in data_cfg:
            kwargs[key] = data_cfg[key]
    return dataset_cls(**_filter_kwargs(dataset_cls, kwargs))


def warmup_vlm_cache(
    config: dict,
    splits: Sequence[str] = ("train",),
    batch_size: int = 1,
    unique_only: bool = False,
    num_workers: int = 4,
    conditioner: Optional[VLMConditioner] = None,
) -> None:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    if not model_cfg.get("use_vlm_film", False):
        print("[VLM-WARMUP] use_vlm_film=False; skipping warmup.")
        return

    if conditioner is None:
        vlm_cfg = model_cfg.get("vlm_film_config", {}) or {}
        conditioner = VLMConditioner(
            enabled=True,
            model_name=vlm_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
            cond_dim=vlm_cfg.get("cond_dim", 256),
            cache_dir=vlm_cfg.get("cache_dir", "cache/vlm_profiles"),
            prompt_template=vlm_cfg.get("prompt_template"),
            dtype=vlm_cfg.get("dtype", "auto"),
            device_map=vlm_cfg.get("device_map", "auto"),
            max_new_tokens=vlm_cfg.get("max_new_tokens", 48),
            pool=vlm_cfg.get("pool", "mean"),
            text_mlp_hidden_dim=vlm_cfg.get("text_mlp_hidden_dim", 256),
            embedding_dtype=vlm_cfg.get("embedding_dtype", "float16"),
            verbose=vlm_cfg.get("verbose", False),
            verbose_debug=vlm_cfg.get("verbose_debug", False),
            log_every_n_steps=vlm_cfg.get("log_every_n_steps", 200),
            vlm_cache_stats_enabled=vlm_cfg.get("vlm_cache_stats_enabled", False),
            vlm_cache_stats_every_n_steps=vlm_cfg.get("vlm_cache_stats_every_n_steps", 200),
        )
    conditioner.eval()

    dataset_info = get_dataset_info(data_cfg.get("name"))
    DataModuleClass = dataset_info.class_ref
    dataset_cls = getattr(DataModuleClass, "dataset_class", DataModuleClass)

    split_map = {
        "train": data_cfg.get("train_dir", dataset_info.default_train_dir),
        "val": data_cfg.get("val_dir", dataset_info.default_val_dir),
        "test": data_cfg.get("test_dir", dataset_info.default_test_dir),
    }

    for split in splits:
        split = str(split).strip()
        if split not in split_map or split_map[split] is None:
            print(f"[VLM-WARMUP] split '{split}' skipped (no path).")
            continue
        ds = _build_dataset(dataset_cls, split_map[split], data_cfg, dataset_info)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        start = time.time()
        print(f"[VLM-WARMUP] split={split} batches={len(dl)} batch_size={batch_size}")
        for step, batch in enumerate(tqdm(dl, desc=f"warmup:{split}")):
            conditioner.warmup_batch(batch, unique_only=unique_only, step=step)
        elapsed = time.time() - start
        stats = getattr(conditioner, "stats", {})
        print(
            f"[VLM-WARMUP] split={split} done: total={stats.get('total', 0)} "
            f"hit={(stats.get('mem_hit', 0)+stats.get('disk_hit', 0))} "
            f"miss={stats.get('miss', 0)} generated={stats.get('generated', 0)} "
            f"elapsed={elapsed:.1f}s"
        )
