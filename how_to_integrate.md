# VLM-FiLM Port Integration (soft-seg behavior)

This repo now includes a direct port of soft-seg's VLM/FiLM conditioning modules. No existing files were modified.
Use the steps below to integrate manually into your model.

## 1) Added files
- `src/archs/components/vlm_conditioner.py`
  - `QwenVLMTextGenerator`, `ImageToConditionVector`, `AdaptiveFiLMHead`, `VLMConditioner`
  - cache + cache key rules, debug payload, mismatch errors
- `src/archs/components/vlm_film.py`
  - FiLM apply logic + logs + env var handling (ported from soft-seg `diffusion_unet.py`)
- `src/archs/components/diffusion_unet_vlm_film.py`
  - `MedSegDiffUNetVLMFiLM` (decoder FiLM hooks)
- `src/archs/components/medsegdiff_flow_vlm_film.py`
  - `medsegdiff_flow_vlm_film`, `medsegdiff_flow_multitask_vlm_film` backbones
- `src/archs/flow_model_vlm_film.py`
  - `FlowModelVLMFiLM` (dfm/cfm with VLM-FiLM)
- `src/conditioning/`
  - `vlm_film_conditioner.py` (compat shim)
  - `__init__.py`, `README.md`, `INTEGRATION_GUIDE.md`, `integration_example.py`, `requirements.txt`
- `src/utils/vlm_cache_warmup.py`
  - VLM cache warmup utility
- `src/runner/train_runner_vlm_film.py`, `src/runner/eval_runner_vlm_film.py`
  - New entrypoints wired to `FlowModelVLMFiLM`
- `scripts/train_vlm_film.py`, `scripts/evaluate_vlm_film.py`
  - CLI scripts using the new runners

## 2) Model init (example)
```python
from src.archs.components.vlm_conditioner import VLMConditioner, AdaptiveFiLMHead

self.use_vlm_film = use_vlm_film
if self.use_vlm_film:
    vlm_cfg = vlm_film_config or {}
    self.vlm_film_conditioner = VLMConditioner(
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
        use_text_prompt_cond=vlm_cfg.get("use_text_prompt_cond", False),
        verbose=vlm_cfg.get("verbose", False),
        verbose_debug=vlm_cfg.get("verbose_debug", False),
        update_interval_steps_train=vlm_cfg.get("update_interval_steps_train", 100),
        update_interval_steps_eval=vlm_cfg.get("update_interval_steps_eval", 1),
        batch_strategy=vlm_cfg.get("batch_strategy", "first"),
        reuse_policy=vlm_cfg.get("reuse_policy", "step_interval"),
        log_every_n_steps=vlm_cfg.get("log_every_n_steps", 200),
        vlm_cache_stats_enabled=vlm_cfg.get("vlm_cache_stats_enabled", False),
        vlm_cache_stats_every_n_steps=vlm_cfg.get("vlm_cache_stats_every_n_steps", 200),
    )

    decoder_channels = [model_channels * m for m in channel_mult][::-1]
    self.vlm_film_heads = nn.ModuleList([
        AdaptiveFiLMHead(
            cond_dim=vlm_cfg.get("cond_dim", 256),
            channels=ch,
            hidden_dim=vlm_cfg.get("text_mlp_hidden_dim", 256),
            gamma_scale=vlm_cfg.get("gamma_scale", 0.1),
            beta_scale=vlm_cfg.get("beta_scale", 0.1),
            use_layernorm=vlm_cfg.get("cond_layernorm", True),
        )
        for ch in decoder_channels
    ])
else:
    self.vlm_film_conditioner = None
    self.vlm_film_heads = None
```

## 3) Compute VLM conditioning
```python
vlm_cond = None
if self.use_vlm_film and self.vlm_film_conditioner is not None:
    vlm_cond = self.vlm_film_conditioner.compute_condition(
        image=images,
        prompt=None,
        image_id=None,
        batch=batch,
        global_step=self.global_step,
        is_train=self.training,
    )
```

## 4) Apply FiLM in decoder (same location as soft-seg)
soft-seg applies FiLM after block2 and before attention.

```python
from src.archs.components.vlm_film import apply_vlm_film

x, _payload = apply_vlm_film(
    x,
    vlm_cond=vlm_cond,
    vlm_film_heads=self.vlm_film_heads,
    stage_idx=stage_idx,
    module=self,
)
```

## 5) Environment variables
- `VLM_FILM_MODE=on|off|shuffle` (default=on)
  - off: gamma=1, beta=0 bypass + identity assert
  - shuffle: batch cond_vec permutation + `[FILM-SHUFFLE]` log
- `VLM_FORCE_NO_ACCELERATE=1`
  - force single-device load even if accelerate exists

## 6) Cache key rules
`VLMConditioner._get_image_ids` uses the exact soft-seg priority order and prints:
- `[VLM] cache_key_example: ...`
- `[VLM] cache_key_source: ...`

## 7) Run training/eval with VLM-FiLM (no existing file edits)
```bash
python scripts/train_vlm_film.py --config <your_config.yaml>
python scripts/evaluate_vlm_film.py --model <model_name> --data <dataset>
```

## 8) Smoke test
```bash
python minimal_smoke_test.py
```
Checks on/off/shuffle behavior, mismatch errors, and `[FILM-WARN]`, `[FILM-VAL]`, `[FILM-SHUFFLE]` logs.

## 9) Dependencies
See `requirements_addition.txt` if you need to add packages.
