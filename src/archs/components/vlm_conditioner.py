"""VLM-to-FiLM conditioning via text generation + text embedding (no JSON)."""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import OrderedDict


# =============================================================================
# Helpers
# =============================================================================

def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


def get_model_input_device(model: nn.Module) -> torch.device:
    """Return a safe device for model inputs (supports device_map models)."""
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in model.hf_device_map.values():
            if dev in ("disk", "cpu", "meta"):
                continue
            return torch.device(dev)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def process_vision_info(messages: List[dict]) -> Tuple[List, List]:
    image_inputs = []
    video_inputs = []
    for msg in messages:
        if "content" in msg:
            for item in msg["content"]:
                if item.get("type") == "image":
                    image_inputs.append(item["image"])
                elif item.get("type") == "video":
                    video_inputs.append(item["video"])
    return image_inputs, video_inputs


def _to_torch_dtype(name: str) -> torch.dtype:
    if isinstance(name, str):
        key = name.lower()
        if key in ("float32", "fp32"):
            return torch.float32
        if key in ("float16", "fp16"):
            return torch.float16
        if key in ("bfloat16", "bf16"):
            return torch.bfloat16
    return torch.float32


# =============================================================================
# Qwen text generator + embedder
# =============================================================================

class QwenVLMTextGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: torch.device | None = None,
        dtype: str = "auto",
        max_new_tokens: int = 48,
        device_map: str | None = "auto",
        verbose: bool = False,
        verbose_debug: bool = False,
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.verbose = verbose
        self.verbose_debug = verbose_debug

        self._model = None
        self._processor = None
        self._using_device_map = False
        self._input_device = None
        self._vlm_loaded = False
        self.last_vision_feat_source = "unknown"
        self.last_vision_feat_shape = None

    def _load_model(self) -> None:
        if self._vlm_loaded:
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("transformers 라이브러리가 필요합니다. pip install transformers")

        if self.verbose:
            print(f"[VLM] Loading {self.model_name}...")

        has_accel = _has_accelerate()
        force_no_accel = os.getenv("VLM_FORCE_NO_ACCELERATE", "0") == "1"
        use_device_map = self.device_map == "auto"
        if use_device_map and (not has_accel or force_no_accel):
            if self.verbose:
                print("[VLM] accelerate not available; falling back to single-device loading.")
            use_device_map = False

        torch_dtype = self.dtype
        if isinstance(torch_dtype, str):
            if torch_dtype.lower() == "auto":
                torch_dtype = "auto"
            else:
                torch_dtype = _to_torch_dtype(torch_dtype)

        if use_device_map:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            self._using_device_map = True
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
            )
            self._model.to(self.device)
            self._using_device_map = False

        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._input_device = get_model_input_device(self._model)
        if self.verbose:
            if self._using_device_map:
                print(f"[VLM] Model loaded with device_map=auto (input_device={self._input_device})")
            else:
                print(f"[VLM] Model loaded on single device: {self._input_device}")
        self._vlm_loaded = True

    def get_text_embed_dim(self) -> int:
        cfg = self._model.config
        text_cfg = getattr(cfg, "text_config", None)
        return int(getattr(text_cfg, "hidden_size", getattr(cfg, "hidden_size", 1024)))

    def get_vision_embed_dim(self) -> int:
        self._load_model()
        cfg = self._model.config
        if hasattr(self._model, "get_image_features"):
            for key in ("image_feature_dim", "image_embed_dim", "projection_dim"):
                val = getattr(cfg, key, None)
                if isinstance(val, int) and val > 0:
                    return int(val)
            text_cfg = getattr(cfg, "text_config", None)
            val = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
            if isinstance(val, int) and val > 0:
                return int(val)
            val = getattr(cfg, "hidden_size", None)
            if isinstance(val, int) and val > 0:
                return int(val)
        vision_cfg = getattr(cfg, "vision_config", None)
        for key in ("projection_dim", "hidden_size", "vision_hidden_size", "embed_dim"):
            val = getattr(vision_cfg, key, None) if vision_cfg is not None else None
            if isinstance(val, int) and val > 0:
                return int(val)
        val = getattr(cfg, "vision_hidden_size", None)
        if isinstance(val, int) and val > 0:
            return int(val)
        val = getattr(cfg, "hidden_size", None)
        if isinstance(val, int) and val > 0:
            return int(val)
        return 1024

    def _coerce_vision_feats(self, feats) -> Optional[torch.Tensor]:
        if isinstance(feats, torch.Tensor):
            return feats
        if isinstance(feats, dict):
            for key in ("pooler_output", "last_hidden_state", "image_embeds"):
                val = feats.get(key)
                if val is None:
                    continue
                coerced = self._coerce_vision_feats(val)
                if coerced is not None:
                    return coerced
            for val in feats.values():
                coerced = self._coerce_vision_feats(val)
                if coerced is not None:
                    return coerced
            return None
        if isinstance(feats, (list, tuple)):
            for item in feats:
                coerced = self._coerce_vision_feats(item)
                if coerced is not None:
                    return coerced
            return None
        for attr in ("pooler_output", "last_hidden_state"):
            if hasattr(feats, attr):
                val = getattr(feats, attr)
                if val is not None:
                    coerced = self._coerce_vision_feats(val)
                    if coerced is not None:
                        return coerced
        return None

    @torch.no_grad()
    def generate_text(self, image: Image.Image, prompt: str) -> str:
        self._load_model()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if not video_inputs:
            video_inputs = None
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        input_device = self._input_device or self.device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        if input_ids is not None:
            generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, output_ids)]
        else:
            generated_ids = output_ids
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if self.verbose_debug:
            print(f"[VLM] Raw output: {output_text[:200]}...")
        return output_text.replace("\n", " ").strip()

    @torch.no_grad()
    def embed_text(self, text: str, pool: str = "mean") -> torch.Tensor:
        self._load_model()
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self._input_device or self.device) for k, v in tokens.items()}
        text_model = getattr(self._model, "language_model", None)
        if text_model is None:
            text_model = getattr(self._model, "model", self._model)
        outputs = text_model(**tokens, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
        mask = tokens.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
        mask = mask.unsqueeze(-1)
        if pool == "last":
            idx = (mask.squeeze(-1).sum(dim=1) - 1).clamp(min=0).long()
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and "input_ids" in tokens:
                input_ids = tokens["input_ids"]
                eos_positions = (input_ids == eos_id).long()
                has_eos = eos_positions.sum(dim=1) > 0
                if has_eos.any():
                    last_eos = eos_positions.cumsum(dim=1).eq(eos_positions.sum(dim=1, keepdim=True))
                    last_eos_idx = last_eos.float().argmax(dim=1)
                    idx = torch.where(has_eos, last_eos_idx, idx)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        else:
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        return pooled

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """Return a global image feature vector from the VLM vision encoder."""
        self._load_model()
        try:
            inputs = self._processor(images=[image], return_tensors="pt")
        except TypeError:
            # Some processors require an explicit text list; provide a stub.
            inputs = self._processor(text=[""], images=[image], return_tensors="pt")
        if not isinstance(inputs, dict) or "pixel_values" not in inputs:
            # Fallback with a minimal text stub if processor requires text
            inputs = self._processor(text=[""], images=[image], return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            for key in ("images", "image"):
                value = inputs.get(key)
                if value is not None:
                    pixel_values = value
                    break
        if pixel_values is None:
            raise RuntimeError("VLM processor did not return pixel_values for vision encoding.")

        input_device = self._input_device or self.device
        pixel_values = pixel_values.to(input_device)
        image_grid_thw = None
        for key in ("image_grid_thw", "grid_thw", "image_grid"):
            value = inputs.get(key)
            if value is not None:
                image_grid_thw = value
                break
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(input_device)

        # Prefer model helper if available (handles grid_thw for Qwen2.5-VL)
        if hasattr(self._model, "get_image_features"):
            try:
                vision_outputs = self._model.get_image_features(
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw
                )
            except TypeError:
                vision_outputs = self._model.get_image_features(pixel_values, image_grid_thw)
        else:
            vision_model = (
                getattr(self._model, "vision_model", None)
                or getattr(self._model, "visual", None)
                or getattr(self._model, "vision_tower", None)
            )
            vision_outputs = None
            if vision_model is not None:
                try:
                    vision_outputs = vision_model(pixel_values=pixel_values, grid_thw=image_grid_thw)
                except TypeError:
                    try:
                        vision_outputs = vision_model(pixel_values, grid_thw=image_grid_thw)
                    except TypeError:
                        vision_outputs = vision_model(pixel_values)
            elif hasattr(self._model, "get_vision_features"):
                vision_outputs = self._model.get_vision_features(pixel_values=pixel_values)
            else:
                raise RuntimeError("Cannot locate a vision encoder on the VLM model.")

        feat_source = "unknown"
        if isinstance(vision_outputs, (list, tuple)):
            # Qwen2.5-VL returns a tuple of per-image embeddings
            feat_list = []
            for f in vision_outputs:
                if isinstance(f, (list, tuple)):
                    f = f[0]
                if f.dim() == 2:
                    feat_list.append(f.mean(dim=0, keepdim=True))
                elif f.dim() == 3:
                    feat_list.append(f.mean(dim=1))
                else:
                    feat_list.append(f.view(1, -1))
            feats = torch.cat(feat_list, dim=0)
            feat_source = "vision_outputs:tuple"
        elif hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            feats = vision_outputs.pooler_output
            feat_source = "vision_outputs:pooler_output"
        elif hasattr(vision_outputs, "last_hidden_state"):
            feats = vision_outputs.last_hidden_state
            feat_source = "vision_outputs:last_hidden_state"
        elif isinstance(vision_outputs, dict):
            feats = (
                vision_outputs.get("pooler_output")
                or vision_outputs.get("last_hidden_state")
                or vision_outputs.get("image_embeds")
            )
            if feats is None:
                feats = next(iter(vision_outputs.values()))
            feat_source = "vision_outputs:dict"
        elif isinstance(vision_outputs, (tuple, list)) and len(vision_outputs) > 0:
            feats = vision_outputs[0]
            feat_source = "vision_outputs:tuple_idx0"
        else:
            feats = vision_outputs
            feat_source = "vision_outputs:tensor"

        self.last_vision_feat_source = feat_source
        if not isinstance(feats, torch.Tensor):
            feats = self._coerce_vision_feats(feats)
            if feats is None:
                raise RuntimeError(
                    "VLM vision features did not resolve to a tensor "
                    f"(source={feat_source} type={type(vision_outputs)})."
                )
        self.last_vision_feat_shape = tuple(feats.shape)

        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.dim() == 3:
            feats = feats.mean(dim=1)
        elif feats.dim() == 2:
            pass
        else:
            feats = feats.view(feats.size(0), -1)

        if isinstance(feats, torch.Tensor):
            self.last_vision_feat_shape = tuple(feats.shape)
        return feats


# =============================================================================
# Text -> Condition Vector -> FiLM
# =============================================================================

class TextToConditionVector(nn.Module):
    def __init__(self, embed_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, e_text: torch.Tensor) -> torch.Tensor:
        return self.mlp(e_text)


class ImageToConditionVector(nn.Module):
    def __init__(self, feat_dim: Optional[int], cond_dim: int, use_layernorm: bool = True):
        super().__init__()
        self.cond_dim = cond_dim
        self.expected_feat_dim = feat_dim if feat_dim is not None and feat_dim > 0 else None
        if feat_dim is None or feat_dim <= 0:
            self.proj = nn.LazyLinear(cond_dim)
        else:
            self.proj = nn.Linear(feat_dim, cond_dim)
        self.ln = nn.LayerNorm(cond_dim) if use_layernorm else None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if isinstance(self.proj, nn.LazyLinear):
            if not self.proj.has_uninitialized_params() and feat.shape[-1] != self.proj.in_features:
                raise RuntimeError(
                    "[VLM] image_to_cond feature dim mismatch: "
                    f"got={feat.shape[-1]} expected={self.proj.in_features} "
                    f"feat_shape={tuple(feat.shape)} expected_feat_dim={self.expected_feat_dim}"
                )
        elif isinstance(self.proj, nn.Linear) and feat.shape[-1] != self.proj.in_features:
            raise RuntimeError(
                "[VLM] image_to_cond feature dim mismatch: "
                f"got={feat.shape[-1]} expected={self.proj.in_features} "
                f"feat_shape={tuple(feat.shape)} expected_feat_dim={self.expected_feat_dim}"
            )
        out = self.proj(feat)
        if self.ln is not None:
            out = self.ln(out)
        return out


class AdaptiveFiLMHead(nn.Module):
    """
    Adaptive Feature-wise Linear Modulation (FiLM) Head.
    
    Converts a global conditioning vector into channel-wise affine parameters
    (gamma, beta) for feature modulation via: x_out = x * gamma + beta.
    
    Design choices:
    1. Identity initialization (zero-init): Starts as identity transformation
       (gamma=1, beta=0), allowing the model to learn from a stable baseline.
    2. No activation constraints (tanh/sigmoid): Allows unbounded modulation,
       relying on junction gating for topology safety in vessel segmentation.
    3. Two-layer MLP: Provides nonlinearity for learning complex conditioning.
    
    Args:
        cond_dim: Dimension of input conditioning vector
        channels: Number of feature channels to modulate
        hidden_dim: Hidden layer size in MLP (default: 256)
        gamma_scale: Deprecated, kept for backward compatibility (default: 0.1)
        beta_scale: Deprecated, kept for backward compatibility (default: 0.1)
        use_layernorm: Deprecated, kept for backward compatibility (default: True)
    
    Forward:
        cond_vec (B, cond_dim) -> (gamma, beta) both (B, channels, 1, 1)
        where gamma = 1 + Linear(c), beta = Linear(c)
    """
    def __init__(
        self,
        cond_dim: int,
        channels: int,
        hidden_dim: int = 256,
        gamma_scale: float = 0.1,  # Kept for backward compat, not used
        beta_scale: float = 0.1,   # Kept for backward compat, not used
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_layernorm = use_layernorm
        
        # Optional LayerNorm on conditioning vector
        if use_layernorm:
            self.ln = nn.LayerNorm(cond_dim)
        
        # Two-layer MLP: cond_dim -> hidden_dim -> channels*2
        # Output: [gamma_offset (channels), beta (channels)]
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels * 2),
        )
        
        # CRITICAL: Zero-initialize final layer for identity at start
        # This ensures: gamma_offset=0 -> gamma=1+0=1, beta=0
        # Result: x_out = x * 1 + 0 = x (identity transformation)
        # Training starts from a stable point and gradually learns modulation
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond_vec: Global conditioning vector (B, cond_dim)
        
        Returns:
            gamma: (B, channels, 1, 1) - Multiplicative modulation, starts at 1
            beta: (B, channels, 1, 1) - Additive modulation, starts at 0
        """
        # Optional LayerNorm
        if self.use_layernorm:
            cond_vec = self.ln(cond_vec)
        
        # MLP output: (B, channels*2)
        params = self.mlp(cond_vec)
        
        # Split into gamma offset and beta
        gamma_offset, beta = params.chunk(2, dim=1)
        
        # gamma = 1 + offset (no constraints, identity at init)
        # beta = raw output (no constraints, zero at init)
        gamma = 1.0 + gamma_offset
        
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        
        return gamma, beta


# =============================================================================
# High-Level Conditioner
# =============================================================================

class VLMConditioner(nn.Module):
    DEFAULT_PROMPT = (
        "Return ONE short sentence describing factors that affect segmentation of the target in this image "
        "(contrast, occlusion, clutter, target size, artifacts). Output ONLY the sentence, one line. "
        "No lists, no JSON, no markdown."
    )

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        cond_dim: int = 256,
        cache_dir: Optional[str] = None,
        prompt_template: Optional[str] = None,
        dtype: str = "auto",
        device_map: str | None = "auto",
        max_new_tokens: int = 48,
        pool: str = "mean",
        text_mlp_hidden_dim: int = 256,
        embedding_dtype: str = "float16",
        use_text_prompt_cond: bool = False,
        verbose: bool = False,
        verbose_debug: bool = False,
        update_interval_steps_train: int = 100,
        update_interval_steps_eval: int = 1,
        batch_strategy: str = "first",
        reuse_policy: str = "step_interval",
        log_every_n_steps: int = 200,
        vlm_cache_stats_enabled: bool = False,
        vlm_cache_stats_every_n_steps: int = 200,
        debug_enabled: bool = True,
        debug_every_n_steps: int = 50,
        mem_cache_size: int = 0,
        film_debug_enabled: bool = True,
        film_debug_every_n_steps_train: int = 100,
        film_debug_every_n_steps_val: int = 20,
    ):
        super().__init__()
        self.enabled = enabled
        self.cond_dim = cond_dim
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.verbose = verbose
        self.verbose_debug = verbose_debug
        self.update_interval_steps_train = int(update_interval_steps_train)
        self.update_interval_steps_eval = int(update_interval_steps_eval)
        self.batch_strategy = str(batch_strategy)
        self.reuse_policy = str(reuse_policy)
        self.log_every_n_steps = int(log_every_n_steps)
        self.cache_stats_enabled = bool(vlm_cache_stats_enabled)
        self.cache_stats_every_n_steps = int(vlm_cache_stats_every_n_steps)
        self.debug_enabled = bool(debug_enabled)
        self.debug_every_n_steps = int(debug_every_n_steps)
        self.mem_cache_size = int(mem_cache_size)
        self._mem_cache: OrderedDict[str, dict] = OrderedDict()
        self.pool = str(pool)
        self.embedding_dtype = _to_torch_dtype(embedding_dtype)
        self.use_text_prompt_cond = bool(use_text_prompt_cond)
        self.film_debug_enabled = bool(film_debug_enabled)
        self.film_debug_every_n_steps_train = int(film_debug_every_n_steps_train)
        self.film_debug_every_n_steps_val = int(film_debug_every_n_steps_val)
        self._last_text = None
        self._last_e_text = None
        self._last_image_feat = None
        self._last_profile_step = -1
        self._last_profile_prompt_hash = None
        self._last_profile_image_key = None
        self._debug_generate_calls = 0
        self.stats = {"total": 0, "mem_hit": 0, "disk_hit": 0, "miss": 0, "generated": 0}
        self._prev_e_text = None
        self._prev_cond = None
        self._prev_cache_key = None
        self._last_key_warn_step = -1000
        self._last_mode = None
        self._last_val_log_step = None

        if self.enabled:
            self.vlm_generator = QwenVLMTextGenerator(
                model_name=model_name,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
                verbose_debug=verbose_debug,
            )
            self.vlm_generator._load_model()
            if self.use_text_prompt_cond:
                embed_dim = self.vlm_generator.get_text_embed_dim()
                self.text_to_cond = TextToConditionVector(
                    embed_dim=embed_dim,
                    cond_dim=cond_dim,
                    hidden_dim=text_mlp_hidden_dim,
                )
            else:
                self.text_to_cond = None
            vision_dim = self.vlm_generator.get_vision_embed_dim()
            self.image_to_cond = ImageToConditionVector(
                feat_dim=vision_dim,
                cond_dim=cond_dim,
                use_layernorm=True,
            )
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.vlm_generator = None
            self.text_to_cond = None
            self.image_to_cond = None

    def _check_image_feat_dim(self, feat_batch: torch.Tensor) -> None:
        if self.image_to_cond is None:
            return
        proj = self.image_to_cond.proj
        expected = None
        if isinstance(proj, nn.Linear):
            expected = proj.in_features
        elif isinstance(proj, nn.LazyLinear) and not proj.has_uninitialized_params():
            expected = proj.in_features
        if expected is None:
            return
        got = feat_batch.shape[-1]
        if got != expected:
            src = getattr(self.vlm_generator, "last_vision_feat_source", "unknown")
            shape = getattr(self.vlm_generator, "last_vision_feat_shape", None)
            raise RuntimeError(
                "[VLM] image_to_cond feature dim mismatch (fixed projection). "
                f"got={got} expected={expected} feat_batch_shape={tuple(feat_batch.shape)} "
                f"vision_embed_dim={self.vlm_generator.get_vision_embed_dim()} "
                f"last_vision_feat_source={src} last_vision_feat_shape={shape}"
            )

    def _normalize_image_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if not isinstance(feat, torch.Tensor):
            return feat
        if feat.dim() == 4:
            return feat.mean(dim=(2, 3))
        if feat.dim() == 3:
            return feat.mean(dim=1)
        if feat.dim() == 2 and feat.size(0) > 1:
            # Treat token-level features from a single image as (T, C) and pool to (1, C).
            return feat.mean(dim=0, keepdim=True)
        if feat.dim() > 2:
            return feat.view(feat.size(0), -1)
        return feat

    def _get_fallback_vision_dim(self) -> int:
        if self.image_to_cond is not None:
            proj = self.image_to_cond.proj
            if isinstance(proj, nn.Linear):
                return int(proj.in_features)
            if isinstance(proj, nn.LazyLinear) and not proj.has_uninitialized_params():
                return int(proj.in_features)
        last_shape = getattr(self.vlm_generator, "last_vision_feat_shape", None)
        if isinstance(last_shape, tuple) and len(last_shape) > 0:
            return int(last_shape[-1])
        return int(self.vlm_generator.get_vision_embed_dim())

    def compute_condition(
        self,
        image: torch.Tensor,
        prompt: Optional[str] = None,
        image_id: Optional[str | List[str]] = None,
        batch: Optional[dict] = None,
        global_step: Optional[int] = None,
        is_train: bool = False,
    ) -> Optional[Dict]:
        if not self.enabled:
            return None

        batch_size = image.shape[0]
        device = image.device

        step = int(global_step) if global_step is not None else -1
        per_sample = self.batch_strategy == "per_sample"
        image_ids, key_source = self._get_image_ids(
            batch=batch,
            images=image,
            step=step,
        )

        final_prompt = prompt or self.prompt_template
        prompt_hash = hashlib.md5(final_prompt.encode()).hexdigest()[:8] if self.use_text_prompt_cond else "img"
        interval = self.update_interval_steps_train if is_train else self.update_interval_steps_eval
        reuse_reason = "generated"

        diag = None
        if per_sample:
            # Per-sample conditioning (no cross-sample reuse in training)
            if self.use_text_prompt_cond:
                e_text_list: List[torch.Tensor] = []
                text_list: List[str] = []
                for i in range(batch_size):
                    sample_id = image_ids[i] if image_ids else None
                    text_i = None
                    e_i = None
                    if sample_id is not None:
                        cached = self._load_from_cache(sample_id, prompt_hash)
                        if cached is not None:
                            text_i = cached.get("text")
                            e_i = cached.get("e_text")
                            if e_i is not None:
                                self.stats["disk_hit"] += 1
                    if e_i is None:
                        pil_image = self._torch_to_pil(image[i])
                        with torch.no_grad():
                            text_i = self.vlm_generator.generate_text(pil_image, final_prompt)
                            e_i = self.vlm_generator.embed_text(text_i, pool=self.pool).to(self.embedding_dtype)
                        self._debug_generate_calls += 1
                        self.stats["generated"] += 1
                        self.stats["miss"] += 1
                        if sample_id is not None:
                            self._save_to_cache(sample_id, prompt_hash, text=text_i, e_text=e_i)
                    if e_i is None:
                        e_i = torch.zeros(self.vlm_generator.get_text_embed_dim(), dtype=self.embedding_dtype)
                        if text_i is None:
                            text_i = ""
                    if e_i.dim() == 2 and e_i.size(0) == 1:
                        e_i = e_i.squeeze(0)
                    if isinstance(e_i, torch.Tensor):
                        e_i = e_i.detach().cpu()
                    e_text_list.append(e_i)
                    text_list.append(text_i or "")

                target_dtype = next(self.text_to_cond.parameters()).dtype
                e_text_batch = torch.stack(e_text_list, dim=0).to(device=device, dtype=target_dtype).detach()
                cond_vec_batch = self.text_to_cond(e_text_batch)
                reuse_reason = "per_sample"
            else:
                image_feat_list: List[torch.Tensor] = []
                text_list: List[str] = []
                for i in range(batch_size):
                    sample_id = image_ids[i] if image_ids else None
                    feat_i = None
                    if sample_id is not None:
                        cached = self._load_from_cache(sample_id, prompt_hash)
                        if cached is not None:
                            feat_i = cached.get("image_feat")
                            if feat_i is not None:
                                self.stats["disk_hit"] += 1
                    if feat_i is None:
                        pil_image = self._torch_to_pil(image[i])
                        with torch.no_grad():
                            feat_i = self.vlm_generator.embed_image(pil_image).to(self.embedding_dtype)
                        self._debug_generate_calls += 1
                        self.stats["generated"] += 1
                        self.stats["miss"] += 1
                        if sample_id is not None:
                            self._save_to_cache(sample_id, prompt_hash, image_feat=feat_i)
                    if feat_i is None:
                        feat_dim = self._get_fallback_vision_dim()
                        feat_i = torch.zeros(feat_dim, dtype=self.embedding_dtype)
                    if isinstance(feat_i, torch.Tensor):
                        feat_i = self._normalize_image_feat(feat_i)
                    if feat_i.dim() == 2 and feat_i.size(0) == 1:
                        feat_i = feat_i.squeeze(0)
                    if isinstance(feat_i, torch.Tensor):
                        feat_i = feat_i.detach().cpu()
                    image_feat_list.append(feat_i)
                    text_list.append("")

                target_dtype = next(self.image_to_cond.parameters()).dtype
                e_text_batch = torch.stack(image_feat_list, dim=0).to(device=device, dtype=target_dtype).detach()
                self._check_image_feat_dim(e_text_batch)
                cond_vec_batch = self.image_to_cond(e_text_batch)
                reuse_reason = "per_sample"
        else:
            # Eval/val path keeps optional reuse policy
            rep_idx = 0
            if self.batch_strategy == "random" and batch_size > 1:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(max(step, 0))
                rep_idx = int(torch.randint(0, batch_size, (1,), generator=gen).item())

            rep_img = image[rep_idx]
            rep_id = image_ids[rep_idx] if image_ids else None

            if self.use_text_prompt_cond:
                text = None
                e_text = None

                can_reuse = self._last_text is not None and self._last_profile_prompt_hash == prompt_hash
                if can_reuse:
                    if self.reuse_policy == "step_interval":
                        if step >= 0 and (step - self._last_profile_step) < interval:
                            text = self._last_text
                            e_text = self._last_e_text
                            reuse_reason = "interval"
                            self.stats["mem_hit"] += batch_size
                    elif self.reuse_policy == "image_id":
                        if (
                            rep_id is not None
                            and rep_id == self._last_profile_image_key
                            and (step < 0 or (step - self._last_profile_step) < interval)
                        ):
                            text = self._last_text
                            e_text = self._last_e_text
                            reuse_reason = "image_id"
                            self.stats["mem_hit"] += batch_size

                if text is None and rep_id is not None:
                    cached = self._load_from_cache(rep_id, prompt_hash)
                    if cached is not None:
                        text = cached.get("text")
                        e_text = cached.get("e_text")
                        if e_text is not None:
                            reuse_reason = "disk_cache"
                            self.stats["disk_hit"] += batch_size

                if text is None or e_text is None:
                    pil_image = self._torch_to_pil(rep_img)
                    with torch.no_grad():
                        text = self.vlm_generator.generate_text(pil_image, final_prompt)
                        e_text = self.vlm_generator.embed_text(text, pool=self.pool).to(self.embedding_dtype)
                    self._debug_generate_calls += 1
                    self.stats["generated"] += batch_size
                    self.stats["miss"] += batch_size
                    if rep_id is not None:
                        self._save_to_cache(rep_id, prompt_hash, text=text, e_text=e_text)

                # Update last cache (text + e_text only, no graph)
                self._last_text = text
                self._last_e_text = e_text.detach()
                if text is not None and e_text is not None:
                    self._last_profile_step = step
                self._last_profile_prompt_hash = prompt_hash
                self._last_profile_image_key = rep_id

                # cond_vec fresh each step (grad-enabled)
                target_dtype = next(self.text_to_cond.parameters()).dtype
                e_text = e_text.to(device=device, dtype=target_dtype).detach()
                if e_text.dim() == 1:
                    e_text = e_text.unsqueeze(0)
                e_text_batch = e_text
                if e_text_batch.size(0) == 1 and batch_size > 1:
                    e_text_batch = e_text_batch.expand(batch_size, -1)
                cond_vec = self.text_to_cond(e_text)
                cond_vec_batch = cond_vec.expand(batch_size, -1).contiguous()
            else:
                image_feat = None

                can_reuse = self._last_image_feat is not None and self._last_profile_prompt_hash == prompt_hash
                if can_reuse:
                    if self.reuse_policy == "step_interval":
                        if step >= 0 and (step - self._last_profile_step) < interval:
                            image_feat = self._last_image_feat
                            reuse_reason = "interval"
                            self.stats["mem_hit"] += batch_size
                    elif self.reuse_policy == "image_id":
                        if (
                            rep_id is not None
                            and rep_id == self._last_profile_image_key
                            and (step < 0 or (step - self._last_profile_step) < interval)
                        ):
                            image_feat = self._last_image_feat
                            reuse_reason = "image_id"
                            self.stats["mem_hit"] += batch_size

                if image_feat is None and rep_id is not None:
                    cached = self._load_from_cache(rep_id, prompt_hash)
                    if cached is not None:
                        image_feat = cached.get("image_feat")
                        if image_feat is not None:
                            reuse_reason = "disk_cache"
                            self.stats["disk_hit"] += batch_size

                if image_feat is None:
                    pil_image = self._torch_to_pil(rep_img)
                    with torch.no_grad():
                        image_feat = self.vlm_generator.embed_image(pil_image).to(self.embedding_dtype)
                    self._debug_generate_calls += 1
                    self.stats["generated"] += batch_size
                    self.stats["miss"] += batch_size
                    if rep_id is not None:
                        self._save_to_cache(rep_id, prompt_hash, image_feat=image_feat)

                if isinstance(image_feat, torch.Tensor):
                    image_feat = self._normalize_image_feat(image_feat)

                # Update last cache (image feat only, no graph)
                self._last_image_feat = image_feat.detach()
                if image_feat is not None:
                    self._last_profile_step = step
                self._last_profile_prompt_hash = prompt_hash
                self._last_profile_image_key = rep_id

                # cond_vec fresh each step (grad-enabled)
                target_dtype = next(self.image_to_cond.parameters()).dtype
                image_feat = image_feat.to(device=device, dtype=target_dtype).detach()
                if image_feat.dim() == 1:
                    image_feat = image_feat.unsqueeze(0)
                e_text_batch = image_feat
                if e_text_batch.size(0) == 1 and batch_size > 1:
                    e_text_batch = e_text_batch.expand(batch_size, -1)
                self._check_image_feat_dim(e_text_batch)
                cond_vec = self.image_to_cond(image_feat)
                cond_vec_batch = cond_vec.expand(batch_size, -1).contiguous()

        # Diagnostics (lightweight, no graph)
        if self.verbose and step >= 0 and self.log_every_n_steps > 0 and (step % self.log_every_n_steps == 0):
            with torch.no_grad():
                e_det = e_text_batch.detach().float()
                c_det = cond_vec_batch.detach().float()
                e_norm = e_det.norm(dim=1).mean().item()
                c_norm = c_det.norm(dim=1).mean().item()
                if c_det.size(0) > 1:
                    c_normed = c_det / (c_det.norm(dim=1, keepdim=True) + 1e-8)
                    sim = c_normed @ c_normed.T
                    bsz = sim.size(0)
                    mask = torch.eye(bsz, device=sim.device, dtype=torch.bool)
                    off = sim[~mask]
                    if off.numel() > 0:
                        cos_mean = off.mean().item()
                        cos_max = off.max().item()
                    else:
                        cos_mean = 1.0
                        cos_max = 1.0
                else:
                    cos_mean = 1.0
                    cos_max = 1.0
            diag = {
                "e_text_norm": e_norm,
                "cond_norm": c_norm,
                "cond_cos_mean": cos_mean,
                "cond_cos_max": cos_max,
            }
            if is_train and self.debug_enabled and step >= 0 and self.debug_every_n_steps > 0 and (step % self.debug_every_n_steps == 0):
                key0 = image_ids[0] if image_ids else "unknown"
                print(f"[VLM-FiLM-DBG] step={step} B={batch_size} key0={key0}")
                if batch_size >= 2:
                    k0 = key0
                    k1 = image_ids[1] if image_ids else "unknown"
                    if "text_list" in locals():
                        r0 = text_list[0][:80]
                        r1 = text_list[1][:80]
                    else:
                        r0 = (self._last_text or "")[:80]
                        r1 = r0
                    same_key = k0 == k1
                    same_resp = r0 == r1
                    e0 = e_det[0]
                    e1 = e_det[1]
                    c0 = c_det[0]
                    c1 = c_det[1]
                    cos_e = F.cosine_similarity(e0, e1, dim=0).item()
                    cos_c = F.cosine_similarity(c0, c1, dim=0).item()
                    print(f"[VLM-FiLM-DBG] k0={k0} k1={k1} same_key={same_key}")
                    print(f"[VLM-FiLM-DBG] r0={r0} r1={r1} same_resp={same_resp}")
                    print(f"[VLM-FiLM-DBG] cos(e0,e1)={cos_e:.4f} cos(c0,c1)={cos_c:.4f}")
                else:
                    print("[VLM-FiLM-DBG] B==1; skip intra-batch compare (trivial).")

                if self._prev_e_text is not None and self._prev_cond is not None:
                    prev_e0 = self._prev_e_text[0].to(e_det.device)
                    prev_c0 = self._prev_cond[0].to(c_det.device)
                    cos_e_step = F.cosine_similarity(e_det[0], prev_e0, dim=0).item()
                    cos_c_step = F.cosine_similarity(c_det[0], prev_c0, dim=0).item()
                    same_key = (key0 == (self._prev_cache_key or ""))
                    print(
                        f"[VLM-FiLM-DBG] step_to_step: same_key={same_key} "
                        f"cos_e={cos_e_step:.4f} cos_cond={cos_c_step:.4f}"
                    )
                self._prev_e_text = e_det.detach().cpu()
                self._prev_cond = c_det.detach().cpu()
                self._prev_cache_key = key0

        self.stats["total"] += batch_size
        if self.cache_stats_enabled and step >= 0 and self.cache_stats_every_n_steps > 0 and (step % self.cache_stats_every_n_steps == 0):
            hit = self.stats["mem_hit"] + self.stats["disk_hit"]
            hit_rate = hit / max(self.stats["total"], 1)
            miss_rate = self.stats["miss"] / max(self.stats["total"], 1)
            print(
                f"[VLM] cache_stats: total={self.stats['total']} hit={hit} "
                f"mem_hit={self.stats['mem_hit']} disk_hit={self.stats['disk_hit']} "
                f"miss={self.stats['miss']} hit_rate={hit_rate:.3f} miss_rate={miss_rate:.3f}"
            )
        if step >= 0 and self.log_every_n_steps > 0 and (step % self.log_every_n_steps == 0):
            print(
                f"[VLM] step={step} train={is_train} reuse={reuse_reason} "
                f"batch_strategy={self.batch_strategy} interval={interval}"
            )
        if step >= 0 and (step % 50 == 0):
            first_key = image_ids[0] if image_ids else "unknown"
            print(f"[VLM] cache_key_example: {first_key}")
            print(f"[VLM] cache_key_source: {key_source}")

        out = {"cond_vec": cond_vec_batch}
        if diag is not None:
            out["diag"] = diag

        dbg_payload = None
        if self.film_debug_enabled and self._is_rank0():
            cond_det = cond_vec_batch.detach().float()
            cond_norm = cond_det.norm(dim=1)
            cond_norm_mean = cond_norm.mean().item()
            cond_norm_std = cond_norm.std(unbiased=False).item() if cond_norm.numel() > 1 else 0.0

            cos_vals = []
            if cond_det.size(0) > 1:
                n_pairs = min(8, cond_det.size(0) * (cond_det.size(0) - 1) // 2)
                gen = torch.Generator(device=cond_det.device)
                gen.manual_seed(max(step, 0))
                idx_a = torch.randint(0, cond_det.size(0), (n_pairs,), generator=gen, device=cond_det.device)
                idx_b = torch.randint(0, cond_det.size(0), (n_pairs,), generator=gen, device=cond_det.device)
                for a, b in zip(idx_a.tolist(), idx_b.tolist()):
                    if a == b:
                        continue
                    cos_vals.append(
                        F.cosine_similarity(cond_det[a], cond_det[b], dim=0).item()
                    )
            if cos_vals:
                cond_cos_mean = float(sum(cos_vals) / len(cos_vals))
                cond_cos_min = float(min(cos_vals))
                cond_cos_max = float(max(cos_vals))
            else:
                cond_cos_mean = 1.0
                cond_cos_min = 1.0
                cond_cos_max = 1.0

            cond_source = "text_prompt" if self.use_text_prompt_cond else "image_global"
            split_key = "unknown"
            mixed_keys = False
            key_example = image_ids[0] if image_ids else "unknown"
            if image_ids:
                prefixes = set()
                for k in image_ids:
                    if isinstance(k, str) and ":" in k:
                        prefixes.add(k.split(":", 1)[0])
                    else:
                        prefixes.add("unknown")
                mixed_keys = len(prefixes) > 1
                split_key = next(iter(prefixes)) if prefixes else "unknown"

            hit = self.stats["mem_hit"] + self.stats["disk_hit"]
            total = max(self.stats["total"], 1)
            hit_rate = hit / total

            should_log = False
            if is_train:
                if step >= 0 and self.film_debug_every_n_steps_train > 0:
                    should_log = (step % self.film_debug_every_n_steps_train == 0)
            else:
                if self._last_val_log_step != step:
                    if self.film_debug_every_n_steps_val > 0:
                        should_log = True
                        self._last_val_log_step = step

            dbg_payload = {
                "stage": "train" if is_train else "val",
                "step": step,
                "B": int(cond_vec_batch.size(0)),
                "split_key": split_key,
                "cache_key_example": key_example,
                "cond_source": cond_source,
                "cond_dim": int(cond_vec_batch.size(1)),
                "cond_norm_mean": cond_norm_mean,
                "cond_norm_std": cond_norm_std,
                "cond_cos_mean": cond_cos_mean,
                "cond_cos_min": cond_cos_min,
                "cond_cos_max": cond_cos_max,
                "cache_total": int(self.stats["total"]),
                "cache_hit": int(hit),
                "cache_miss": int(self.stats["miss"]),
                "hit_rate": float(hit_rate),
                "mem_hit": int(self.stats["mem_hit"]),
                "disk_hit": int(self.stats["disk_hit"]),
                "mixed_keys": mixed_keys,
                "should_log": bool(should_log),
            }
            out["_dbg"] = dbg_payload
        return out

    @torch.inference_mode()
    def warmup_batch(
        self,
        batch: dict,
        prompt: Optional[str] = None,
        unique_only: bool = False,
        step: Optional[int] = None,
    ) -> None:
        """Populate disk cache for a batch (no gradients, no FiLM computation)."""
        if not self.enabled:
            return
        if not isinstance(batch, dict) or "image" not in batch:
            return

        images = batch["image"]
        if not torch.is_tensor(images):
            return

        batch_size = int(images.shape[0])
        warm_step = int(step) if step is not None else 0
        image_ids, key_source = self._get_image_ids(batch=batch, images=images, step=warm_step)

        final_prompt = prompt or self.prompt_template
        prompt_hash = hashlib.md5(final_prompt.encode()).hexdigest()[:8] if self.use_text_prompt_cond else "img"

        if unique_only:
            if not hasattr(self, "_warmup_seen"):
                self._warmup_seen = set()
            seen = self._warmup_seen
        else:
            seen = None

        for i in range(batch_size):
            sample_id = image_ids[i] if image_ids else None
            if sample_id is None:
                continue
            if seen is not None:
                if sample_id in seen:
                    continue
                seen.add(sample_id)
            cached = self._load_from_cache(sample_id, prompt_hash)
            if cached is not None:
                has_payload = False
                if self.use_text_prompt_cond:
                    has_payload = cached.get("e_text") is not None
                else:
                    has_payload = cached.get("image_feat") is not None
                if has_payload:
                    self.stats["disk_hit"] += 1
                    self.stats["total"] += 1
                    continue

            pil_image = self._torch_to_pil(images[i])
            if self.use_text_prompt_cond:
                text_i = self.vlm_generator.generate_text(pil_image, final_prompt)
                e_i = self.vlm_generator.embed_text(text_i, pool=self.pool).to(self.embedding_dtype)
                self._save_to_cache(sample_id, prompt_hash, text=text_i, e_text=e_i)
            else:
                feat_i = self.vlm_generator.embed_image(pil_image).to(self.embedding_dtype)
                self._save_to_cache(sample_id, prompt_hash, image_feat=feat_i)
            self.stats["generated"] += 1
            self.stats["miss"] += 1
            self.stats["total"] += 1

        if self.verbose and self.log_every_n_steps > 0 and (warm_step % self.log_every_n_steps == 0):
            first_key = image_ids[0] if image_ids else "unknown"
            print(f"[VLM] cache_key_example: {first_key}")
            print(f"[VLM] cache_key_source: {key_source}")

    def forward(self, images: torch.Tensor, train: bool, step: Optional[int] = None, batch: Optional[dict] = None) -> torch.Tensor:
        cond = self.compute_condition(
            image=images,
            prompt=None,
            image_id=None,
            batch=batch,
            global_step=step,
            is_train=train,
        )
        if cond is None:
            return None
        return cond.get("cond_vec")

    def _get_image_ids(
        self,
        batch: Optional[dict],
        images: torch.Tensor,
        step: int,
    ) -> tuple[list[str], str]:
        batch_size = int(images.shape[0])
        split = "unknown"
        if isinstance(batch, dict):
            split = batch.get("split") or batch.get("stage") or batch.get("mode") or split
        split = str(split)

        def _to_list(value, bsz: int):
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                if torch.is_tensor(value) and value.ndim == 1 and value.numel() == bsz:
                    return [v.item() for v in value]
            except Exception:
                pass
            try:
                if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == bsz:
                    return list(value)
            except Exception:
                pass
            return None

        def _normalize_list(items, dataset_root):
            keys = []
            for item in items:
                key = self._normalize_key(item, dataset_root)
                if not key:
                    keys.append(f"{split}:unknown")
                else:
                    keys.append(f"{split}:{key}")
            return keys

        source = "fallback"
        dataset_root = None
        scalar_meta = False

        def _pick_from_batch_keys(keys, bsz):
            if not isinstance(batch, dict):
                return None, None
            for key in keys:
                if key in batch:
                    items = _to_list(batch.get(key), bsz)
                    if items is not None and len(items) == bsz:
                        return items, f"batch[{key}]"
            return None, None

        def _pick_from_meta_dict(md, bsz):
            if not isinstance(md, dict):
                return None
            v = md.get("filename_or_obj") or md.get("filepath") or md.get("path") or md.get("filename") or md.get("image_id")
            items = _to_list(v, bsz)
            if items is not None and len(items) == bsz:
                return items
            nested = md.get("meta")
            if isinstance(nested, dict):
                v2 = nested.get("filename_or_obj") or nested.get("filepath") or nested.get("path") or nested.get("filename") or nested.get("image_id")
                items2 = _to_list(v2, bsz)
                if items2 is not None and len(items2) == bsz:
                    return items2
            return None

        # A) batch["name"] (file-based key)
        if isinstance(batch, dict) and "name" in batch:
            raw = batch.get("name")
            items = _to_list(raw, batch_size)
            if items is None and isinstance(raw, str):
                items = [raw] * batch_size
            if items is not None and len(items) == batch_size:
                norm_items = []
                for item in items:
                    s = str(item)
                    name_base = s.split("@", 1)[0]
                    norm_items.append(name_base)
                return _normalize_list(norm_items, dataset_root), "batch[name]"

        # B) image_meta_dict from list_data_collate
        if isinstance(batch, dict) and "image_meta_dict" in batch:
            meta = batch.get("image_meta_dict")
            if isinstance(meta, dict):
                dataset_root = meta.get("dataset_root") or batch.get("dataset_root")
                items = _pick_from_meta_dict(meta, batch_size)
                if items is not None:
                    return _normalize_list(items, dataset_root), "image_meta_dict"
                if meta.get("filename_or_obj") is not None and batch_size > 1:
                    scalar_meta = True

        # C) meta_dict or similar common MONAI keys (batch-level)
        items, src = _pick_from_batch_keys(
            ("meta_dict", "metadata", "meta", "image_path", "path", "image_id", "id", "filename"),
            batch_size,
        )
        if items is not None:
            return _normalize_list(items, dataset_root), src

        # D) MetaTensor meta
        if isinstance(batch, dict) and "image" in batch:
            img_obj = batch.get("image")
            if hasattr(img_obj, "meta") and isinstance(img_obj.meta, dict):
                dataset_root = img_obj.meta.get("dataset_root") or dataset_root
                items = _pick_from_meta_dict(img_obj.meta, batch_size)
                if items is not None:
                    return _normalize_list(items, dataset_root), "meta_tensor"
                if img_obj.meta.get("filename_or_obj") is not None and batch_size > 1:
                    scalar_meta = True

        # E) Alternative batch keys (paths/ids)
        items, src = _pick_from_batch_keys(
            ("filepath", "path", "image_id", "id", "filename", "index", "idx", "case_id"),
            batch_size,
        )
        if items is not None:
            return _normalize_list(items, dataset_root), src

        # F) last resort: deterministic per-item keys (no step)
        if scalar_meta and self.verbose and step >= 0 and (step - self._last_key_warn_step) >= 100:
            print("[VLM] Warning: scalar filename_or_obj with B>1; using synthetic cache keys.")
            self._last_key_warn_step = step

        # Try fingerprint from per-sample metadata if available
        meta = None
        if isinstance(batch, dict) and "image" in batch:
            img_obj = batch.get("image")
            if hasattr(img_obj, "meta") and isinstance(img_obj.meta, dict):
                meta = img_obj.meta

        def _per_sample_meta(m, key, bsz):
            if not isinstance(m, dict) or key not in m:
                return None
            v = m.get(key)
            lst = _to_list(v, bsz)
            if lst is not None and len(lst) == bsz:
                return lst
            return None

        patch_idx = _per_sample_meta(meta, "patch_index", batch_size)
        crop_ctr = _per_sample_meta(meta, "crop_center", batch_size)
        spatial = _per_sample_meta(meta, "spatial_shape", batch_size)

        if patch_idx or crop_ctr or spatial:
            items = []
            for i in range(batch_size):
                token = f"{patch_idx[i] if patch_idx else ''}|{crop_ctr[i] if crop_ctr else ''}|{spatial[i] if spatial else ''}"
                fp = hashlib.md5(str(token).encode()).hexdigest()[:12]
                items.append(f"unknown:fp={fp}:i={i}")
            return _normalize_list(items, dataset_root), "fallback_fp_meta"

        # Fallback: fast deterministic fingerprint from tensor bytes
        items = []
        for i in range(batch_size):
            x = torch.as_tensor(images[i].detach().cpu()).contiguous()
            byte_view = x.view(torch.uint8)
            n = byte_view.numel()
            if n == 0:
                fp = "empty"
            else:
                chunk = 256
                if n < chunk:
                    sample_bytes = byte_view.cpu().numpy().tobytes()
                else:
                    mid = max(0, n // 2 - chunk // 2)
                    sample_bytes = (
                        byte_view[:chunk].cpu().numpy().tobytes()
                        + byte_view[mid:mid + chunk].cpu().numpy().tobytes()
                        + byte_view[-chunk:].cpu().numpy().tobytes()
                    )
                header = f"{tuple(x.shape)}|{x.dtype}".encode()
                fp = hashlib.md5(header + sample_bytes).hexdigest()[:12]
            items.append(f"unknown:fp={fp}:i={i}")
        return _normalize_list(items, dataset_root), "fallback_fp_tensor"

    @staticmethod
    def _normalize_key(src, dataset_root: Optional[str]) -> str:
        if src is None:
            return ""
        if hasattr(src, "name") and not isinstance(src, (str, bytes)):
            src = getattr(src, "name")
        try:
            import numpy as _np
            if isinstance(src, _np.generic):
                src = src.item()
        except Exception:
            pass
        src = str(src)
        delimiters = ["::", "|", "#", "?"]
        cut = len(src)
        for d in delimiters:
            idx = src.find(d)
            if idx != -1:
                cut = min(cut, idx)
        src = src[:cut].strip()
        if dataset_root:
            try:
                root = str(dataset_root)
                if src.startswith(root):
                    src = os.path.relpath(src, root)
            except Exception:
                pass
        return src

    def _get_cache_path(self, image_id: str, prompt_hash: str) -> Path:
        filename = f"{image_id}_{prompt_hash}.pt"
        return self.cache_dir / filename

    def _load_from_cache(self, image_id: str, prompt_hash: str) -> Optional[dict]:
        if not self.cache_dir:
            return None
        if self.mem_cache_size > 0:
            key = f"{image_id}_{prompt_hash}"
            cached = self._mem_cache.get(key)
            if cached is not None:
                self._mem_cache.move_to_end(key)
                return cached
        cache_path = self._get_cache_path(image_id, prompt_hash)
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                if self.mem_cache_size > 0:
                    key = f"{image_id}_{prompt_hash}"
                    self._mem_cache[key] = payload
                    self._mem_cache.move_to_end(key)
                    if len(self._mem_cache) > self.mem_cache_size:
                        self._mem_cache.popitem(last=False)
                return payload
            except Exception:
                return None
        return None

    def _save_to_cache(
        self,
        image_id: str,
        prompt_hash: str,
        text: str | None = None,
        e_text: torch.Tensor | None = None,
        image_feat: torch.Tensor | None = None,
    ) -> None:
        if not self.cache_dir:
            return
        cache_path = self._get_cache_path(image_id, prompt_hash)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {}
            if text is not None:
                payload["text"] = text
            if e_text is not None:
                payload["e_text"] = e_text.detach().cpu()
            if image_feat is not None:
                payload["image_feat"] = image_feat.detach().cpu()
            if not payload:
                return
            tmp_path = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
            if self.mem_cache_size > 0:
                key = f"{image_id}_{prompt_hash}"
                self._mem_cache[key] = payload
                self._mem_cache.move_to_end(key)
                if len(self._mem_cache) > self.mem_cache_size:
                    self._mem_cache.popitem(last=False)
        except Exception:
            if self.verbose:
                print("[VLM] cache write failed")

    def _torch_to_pil(self, img: torch.Tensor) -> Image.Image:
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        return Image.fromarray(img_np)

    def get_debug_stats(self) -> dict:
        return {"vlm_generate_calls": int(self._debug_generate_calls)}

    @staticmethod
    def _is_rank0() -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True


def debug_key_consistency() -> None:
    """Lightweight check: same image tensor -> same keys across steps."""
    cond = VLMConditioner(enabled=False, verbose=False)
    images = torch.zeros(2, 1, 8, 8)
    keys1, _ = cond._get_image_ids(batch=None, images=images, step=0)
    keys2, _ = cond._get_image_ids(batch=None, images=images, step=5)
    print("[VLM-DBG] keys step0:", keys1)
    print("[VLM-DBG] keys step5:", keys2)
    assert keys1 == keys2, "Keys must be step-invariant"


if __name__ == "__main__":  # pragma: no cover
    debug_key_consistency()
