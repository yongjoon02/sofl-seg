"""Flow matching models for vessel segmentation."""
import autorootcwd  # noqa: F401
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as L
from monai.inferers import SlidingWindowInferer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.segmentation.dice import DiceScore
from torchdiffeq import odeint

from src.registry.base import ARCHS_REGISTRY
from src.archs.components import unet  # noqa: F401 - Register architectures
from src.archs.components import medsegdiff_flow_vlm_film  # noqa: F401 - Register VLM-FiLM backbones
from src.archs.components.flow import SchrodingerBridgeConditionalFlowMatcher
from src.archs.components.vlm_conditioner import VLMConditioner, AdaptiveFiLMHead
from src.archs.dfm_binary import (
    loss_cfm,
    loss_dfm_binary,
    make_xt_binary,
    make_xt_continuous,
    sampler_dfm_euler,
    sampler_dfm_heun,
    _soft_dice_loss,
    to_binary_mask,
)
from src.metrics.vessel_metrics import clDice, Betti0Error, Betti1Error
from src.archs.components.utils import random_patch_batch, select_patch_params


class _LossSummaryModule(nn.Module):
    """Lightweight module to expose loss configuration in Lightning logs."""

    def __init__(self, description: str) -> None:
        super().__init__()
        self.description = description

    def forward(self, *args, **kwargs):  # pragma: no cover - not meant to be called
        raise RuntimeError("Loss summary module is not callable.")

    def extra_repr(self) -> str:
        return self.description

class FlowModelVLMFiLM(L.LightningModule):
    """Lightning module for flow matching producing binary masks."""
    
    def __init__(
        self,
        arch_name: str = 'dhariwal_concat_unet',
        image_size: int = 512,
        patch_plan: list = [(320, 6), (384, 4), (416, 3), (512, 1)],
        dim: int = 32,
        timesteps: int = 15,
        sigma: float = 0.25,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        experiment_name: str = None,
        num_ensemble: int = 1,
        data_name: str = 'xca',
        log_image_enabled: bool = False,
        log_image_names: list = None,
        use_sliding_infer: bool = True,
        # UNet architecture parameters
        model_channels: int = 32,
        channel_mult: list = [1, 2, 4, 8],
        channel_mult_emb: int = 4,
        num_blocks: int = 3,
        attn_resolutions: list = [16, 16, 8, 8],
        dropout: float = 0.0,
        label_dim: int = 0,
        augment_dim: int = 0,
        use_gradient_checkpointing: bool = False,
        # Loss configuration
        loss_type: str = 'l2',  # 'l2', 'l2_bce', 'l2_l2', 'l2_bce_l2', 'l2_bce_dice' (also supports legacy 'l1*')
        bce_weight: float = 0.5,
        l2_weight: float = 0.1,
        dice_weight: float = 0.2,
        lambda_soft: float = 1.0,
        loss: dict | None = None,
        # Flow matching mode
        mode: str = 'cfm_continuous',
        dfm_sampler: str = 'euler',
        dfm_eps: float = 1e-6,
        debug_dfm: bool = False,
        # VLM-FiLM conditioning (soft-seg port)
        use_vlm_film: bool = False,
        vlm_film_config: dict | None = None,
        vlm_film_decoder_stages: list | None = None,  # e.g., [0,1] or [2,3]. None = [0,1] (Stage 4/3 only)
        vlm_update_interval: int = 50,
        vlm_update_interval_eval: int = 1,
        # Junction-aware FiLM gating (topology safety gate)
        junction_gating_config: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"{data_name}/{arch_name}"
        self.experiment_name = experiment_name
        self.data_name = data_name

        # VLM-FiLM toggle
        self.use_vlm_film = bool(use_vlm_film)
        arch_name_for_unet = arch_name
        if self.use_vlm_film:
            if arch_name == 'medsegdiff_flow':
                arch_name_for_unet = 'medsegdiff_flow_vlm_film'
            elif arch_name == 'medsegdiff_flow_multitask':
                arch_name_for_unet = 'medsegdiff_flow_multitask_vlm_film'
            elif arch_name in ('medsegdiff_flow_vlm_film', 'medsegdiff_flow_multitask_vlm_film'):
                arch_name_for_unet = arch_name
            else:
                raise ValueError("VLM-FiLM is only supported for medsegdiff_flow backbones.")
        
        # Get model class from registry
        if arch_name_for_unet not in ARCHS_REGISTRY:
            raise ValueError(f"Unknown architecture: {arch_name_for_unet}. Available: {list(ARCHS_REGISTRY.keys())}")
        
        arch_class = ARCHS_REGISTRY.get(arch_name_for_unet)

        self.is_multitask = arch_name in ('medsegdiff_flow_multitask', 'medsegdiff_flow_multitask_vlm_film')
        
        self.use_geometry_head = arch_name == 'dhariwal_concat_unet_multihead'

        self.mode = mode
        if self.mode not in {'cfm_continuous', 'dfm_binary'}:
            raise ValueError(f"Invalid flow mode '{self.mode}'.")
        if self.mode == 'dfm_binary' and self.is_multitask:
            raise ValueError("dfm_binary does not support multitask flow models.")
        self.dfm_sampler = dfm_sampler
        if self.dfm_sampler not in {'euler', 'heun'}:
            raise ValueError(f"Invalid dfm_sampler '{self.dfm_sampler}'.")

        # For dhariwal_concat_unet, need mask_channels and input_img_channels
        arch_kwargs = {}
        if arch_name_for_unet in {
            'medsegdiff_flow',
            'medsegdiff_flow_multitask',
            'medsegdiff_flow_vlm_film',
            'medsegdiff_flow_multitask_vlm_film',
        }:
            arch_kwargs['use_gradient_checkpointing'] = use_gradient_checkpointing

        if arch_name in {'dhariwal_concat_unet', 'dhariwal_concat_unet_multihead'}:
            self.unet = arch_class(
                img_resolution=image_size,
                mask_channels=1,  # geometry output
                input_img_channels=1,  # image condition
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dim=label_dim,
                augment_dim=augment_dim,
                **arch_kwargs,
            )
        else:
            self.unet = arch_class(
                img_resolution=image_size,
                model_channels=model_channels,
                channel_mult=channel_mult,
                channel_mult_emb=channel_mult_emb,
                num_blocks=num_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                label_dim=label_dim,
                augment_dim=augment_dim,
                **arch_kwargs,
            )

        # VLM-FiLM Conditioning (soft-seg port)
        self._vlm_update_interval = int(vlm_update_interval)
        self._vlm_update_interval_eval = int(vlm_update_interval_eval)
        self._vlm_film_last_step = -1
        self._vlm_film_grad_checked = False
        if self.use_vlm_film:
            vlm_config = vlm_film_config or {}
            print("[VLM-FiLM] Initializing Qwen2.5-VL-3B-Instruct based conditioning...")
            self.vlm_film_conditioner = VLMConditioner(
                enabled=True,
                model_name=vlm_config.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
                cond_dim=vlm_config.get("cond_dim", 256),
                cache_dir=vlm_config.get("cache_dir", "cache/vlm_profiles"),
                prompt_template=vlm_config.get("prompt_template"),
                dtype=vlm_config.get("dtype", "auto"),
                device_map=vlm_config.get("device_map", "auto"),
                max_new_tokens=vlm_config.get("max_new_tokens", 48),
                pool=vlm_config.get("pool", "mean"),
                text_mlp_hidden_dim=vlm_config.get("text_mlp_hidden_dim", 256),
                embedding_dtype=vlm_config.get("embedding_dtype", "float16"),
                use_text_prompt_cond=vlm_config.get("use_text_prompt_cond", False),
                verbose=vlm_config.get("verbose", False),
                verbose_debug=vlm_config.get("verbose_debug", False),
                update_interval_steps_train=vlm_config.get("update_interval_steps_train", 100),
                update_interval_steps_eval=vlm_config.get("update_interval_steps_eval", 1),
                batch_strategy=vlm_config.get("batch_strategy", "first"),
                reuse_policy=vlm_config.get("reuse_policy", "step_interval"),
                log_every_n_steps=vlm_config.get("log_every_n_steps", 200),
                vlm_cache_stats_enabled=vlm_config.get("vlm_cache_stats_enabled", False),
                vlm_cache_stats_every_n_steps=vlm_config.get("vlm_cache_stats_every_n_steps", 200),
            )
            decoder_channels = [model_channels * m for m in channel_mult][::-1]
            
            # Select decoder stages to apply VLM-FiLM
            if vlm_film_decoder_stages is not None:
                selected_stages = sorted(vlm_film_decoder_stages)
                selected_channels = [decoder_channels[i] for i in selected_stages if i < len(decoder_channels)]
                self._vlm_film_stage_indices = selected_stages
                print(f"[VLM-FiLM] Applying to decoder stages: {selected_stages}")
            else:
                # Default: Stage 4/3 only (indices 0, 1)
                selected_stages = [0, 1]
                selected_channels = decoder_channels[:2]
                self._vlm_film_stage_indices = selected_stages
                print(f"[VLM-FiLM] Applying to default stages (Stage 4/3): {selected_stages}")
            
            self.vlm_film_heads = nn.ModuleList([
                AdaptiveFiLMHead(
                    cond_dim=vlm_config.get("cond_dim", 256),
                    channels=ch,
                    hidden_dim=vlm_config.get("text_mlp_hidden_dim", 256),
                    gamma_scale=vlm_config.get("gamma_scale", 0.1),
                    beta_scale=vlm_config.get("beta_scale", 0.1),
                    use_layernorm=vlm_config.get("cond_layernorm", True),
                )
                for ch in selected_channels
            ])
            print(f"[VLM-FiLM] Created {len(self.vlm_film_heads)} FiLM heads for channels: {selected_channels}")
            for i, (stage_idx, ch) in enumerate(zip(self._vlm_film_stage_indices, selected_channels)):
                print(f"[VLM-FiLM]   Head {i} -> Stage {stage_idx}: {ch} channels")
            disabled_stages = [i for i in range(len(decoder_channels)) if i not in self._vlm_film_stage_indices]
            if disabled_stages:
                print(f"[VLM-FiLM] Disabled stages: {disabled_stages}")
            print(f"[VLM-FiLM] Cache directory: {vlm_config.get('cache_dir', 'cache/vlm_profiles')}")
            print("[VLM-FiLM] ✓ Initialization complete")
        else:
            self.vlm_film_conditioner = None
            self.vlm_film_heads = None
            self._vlm_film_stage_indices = None

        # Junction-aware FiLM gating configuration
        self.junction_gating_config = junction_gating_config
        if self.junction_gating_config is not None and self.junction_gating_config.get('enabled', False):
            print("[Junction Gating] ✓ Enabled")
            warmup_epochs = self.junction_gating_config.get('warmup_epochs', None)
            if warmup_epochs is not None:
                print(f"[Junction Gating]   warmup_epochs: {warmup_epochs} (gates=1.0 during warm-up)")
            print(f"[Junction Gating]   source: {self.junction_gating_config.get('source', 'pred')}")
            print(f"[Junction Gating]   threshold: {self.junction_gating_config.get('threshold', 0.5)}")
            print(f"[Junction Gating]   degree_threshold: {self.junction_gating_config.get('degree_threshold', 3)}")
            print(f"[Junction Gating]   radius_px: {self.junction_gating_config.get('radius_px', 8)}")
            print(f"[Junction Gating]   gate_value_in_junction: {self.junction_gating_config.get('gate_value_in_junction', 0.0)}")
            print(f"[Junction Gating]   apply_stages: {self.junction_gating_config.get('apply_stages', 'same_as_film')}")
        else:
            print("[Junction Gating] ✗ Disabled (default)")

        self.flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        
        # Metrics
        self.val_metrics = MetricCollection({
            'dice': DiceScore(num_classes=num_classes, average='macro', include_background=False, input_format="index"),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'specificity': BinarySpecificity(),
            'iou': BinaryJaccardIndex(),
        })
        
        self.vessel_metrics = MetricCollection({
            'cldice': clDice(),
            'betti_0_error': Betti0Error(),
            'betti_1_error': Betti1Error(),
        })

        self.eval_save_intermediate = False
        self.eval_intermediate_dir = None
        self.eval_intermediate_t = None
        self.eval_intermediate_steps = None
        self.eval_intermediate_max_samples = 0
        self._eval_intermediate_saved = set()
        
        self.log_image_enabled = log_image_enabled
        self.log_image_names = log_image_names if log_image_names is not None else ['00036.png']
        self.use_sliding_infer = use_sliding_infer
        
        # Loss configuration (registry 우선)
        from src.registry import LOSS_REGISTRY

        self.use_registry_loss = loss is not None
        if self.use_registry_loss:
            loss_name = loss.get('name')
            loss_params = loss.get('params', {})
            if loss_name not in LOSS_REGISTRY:
                raise ValueError(f"Unknown loss: {loss_name}. Available: {LOSS_REGISTRY.keys()}")
            self.loss_fn = LOSS_REGISTRY.get(loss_name)(**loss_params)
            self.loss_description = f"{loss_name}({loss_params})"
        else:
            self.loss_fn = None
            self.loss_type = loss_type
            self.bce_weight = bce_weight
            self.l2_weight = l2_weight
            self.dice_weight = dice_weight
            if 'dice' in loss_type:
                from src.losses import DiceLoss
                self.dice_loss = DiceLoss()
            self.loss_description = f"builtin:{loss_type}"

        # Register lightweight summary module so logs show loss info.
        self.loss_summary = _LossSummaryModule(self.loss_description)

        # Buffer for distributed image logging (validation/test).
        self._pending_image_logs: list[dict] = []

        # Sliding window inferer for validation/test (match diffusion behavior).
        self.inferer = SlidingWindowInferer(
            roi_size=(image_size, image_size),
            sw_batch_size=4,
            overlap=0.25,
            mode='gaussian',
        )

    def _subset_batch_for_vlm(self, batch: dict | None, idx: torch.Tensor) -> dict | None:
        if batch is None or not isinstance(batch, dict):
            return None
        try:
            bsz = int(batch["image"].shape[0])
        except Exception:
            return None
        if idx is None:
            return batch
        if isinstance(idx, torch.Tensor):
            idx_tensor = idx
            idx_list = idx.detach().cpu().tolist()
        else:
            idx_list = list(idx)
            idx_tensor = torch.as_tensor(idx_list, device=batch["image"].device)

        def _subset_value(value):
            try:
                if torch.is_tensor(value) and value.shape[0] == bsz:
                    return value.index_select(0, idx_tensor.to(value.device))
            except Exception:
                pass
            if isinstance(value, (list, tuple)) and len(value) == bsz:
                return [value[i] for i in idx_list]
            return value

        out = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                nested = {}
                for n_key, n_val in value.items():
                    nested[n_key] = _subset_value(n_val)
                out[key] = nested
            else:
                out[key] = _subset_value(value)
        out["image"] = batch["image"].index_select(0, idx_tensor.to(batch["image"].device))
        return out

    def _get_vlm_film_cond(self, images: torch.Tensor, batch: dict | None = None) -> dict | None:
        if not self.use_vlm_film or self.vlm_film_conditioner is None:
            return None
        vlm_cond = self.vlm_film_conditioner.compute_condition(
            image=images,
            prompt=None,
            image_id=None,
            batch=batch,
            global_step=self.global_step,
            is_train=self.training,
        )
        # Add stage indices to vlm_cond for decoder routing
        if vlm_cond is not None and self._vlm_film_stage_indices is not None:
            vlm_cond['_vlm_film_stage_indices'] = self._vlm_film_stage_indices
            if not hasattr(self, '_logged_stage_indices'):
                print(f"[_get_vlm_film_cond] Setting _vlm_film_stage_indices={self._vlm_film_stage_indices} in vlm_cond")
                self._logged_stage_indices = True
        return vlm_cond

    def _unet_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: torch.Tensor,
        batch: dict | None = None,
        vlm_cond: dict | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_vlm_film or self.vlm_film_conditioner is None or self.vlm_film_heads is None:
            return self.unet(x, t, images)
        if vlm_cond is None:
            vlm_cond = self._get_vlm_film_cond(images, batch)
        
        # Add junction gating config to vlm_cond if enabled
        if vlm_cond is not None and hasattr(self, 'junction_gating_config') and self.junction_gating_config is not None:
            vlm_cond['junction_gating_config'] = self.junction_gating_config
        
        return self.unet(
            x,
            t,
            images,
            vlm_cond=vlm_cond,
            vlm_film_heads=self.vlm_film_heads,
        )

    def _infer_sliding(self, images: torch.Tensor) -> torch.Tensor:
        """Run sliding-window inference with consistent noise per image."""
        if self.mode == 'dfm_binary':
            return self._infer_sliding_dfm(images)
        if self.is_multitask:
            noise_full = self._sample_x0_like(images, channels=self._infer_noise_channels())
            combined = torch.cat([images, noise_full], dim=1)

            def _sample_fn(x: torch.Tensor) -> torch.Tensor:
                imgs = x[:, :1, :, :]
                noise = x[:, 1:, :, :]
                output_geometry = self.sample(noise, imgs)
                return self._geometry_to_prob(output_geometry)

            prob = self.inferer(combined, _sample_fn)
            return self._prob_to_geometry(prob)

        noise_full = self._sample_x0_like(images, channels=self._infer_noise_channels())

        def _sample_fn(x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
            # Use per-image noise cropped to each patch (condition is sliced by the inferer).
            noise = condition if condition is not None else torch.randn_like(x)
            output_geometry = self.sample(noise, x)
            # Merge in probability space to avoid artifacts at patch boundaries.
            return self._geometry_to_prob(output_geometry)

        prob = self.inferer(images, _sample_fn, condition=noise_full)
        return self._prob_to_geometry(prob)

    def _infer_full(self, images: torch.Tensor) -> torch.Tensor:
        """Run full-image inference (no sliding window)."""
        if self.mode == 'dfm_binary':
            return self.sample(None, images)
        noise = self._sample_x0_like(images, channels=self._infer_noise_channels())
        return self.sample(noise, images)

    def _infer_sliding_dfm(self, images: torch.Tensor) -> torch.Tensor:
        """Sliding-window inference for DFM (no initial noise)."""
        return self._sliding_window_predict(images)

    def _sliding_window_predict(self, images: torch.Tensor) -> torch.Tensor:
        """Sliding-window predict that clamps patch outputs to [0, 1]."""
        def _sample_fn(x: torch.Tensor) -> torch.Tensor:
            output = self.sample(None, x)
            return torch.clamp(output, 0.0, 1.0)

        return self.inferer(images, _sample_fn)

    def _infer_noise_channels(self) -> int:
        return 2 if self.is_multitask else 1

    @staticmethod
    def _geometry_to_prob(geometry: torch.Tensor) -> torch.Tensor:
        """Map geometry in [-1, 1] to probability [0, 1] with clipping."""
        return torch.clamp((geometry + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _prob_to_geometry(prob: torch.Tensor) -> torch.Tensor:
        """Map probability [0, 1] back to geometry [-1, 1]."""
        return prob * 2.0 - 1.0

    def _parse_sample_name(self, sample_name: str) -> str:
        name = str(sample_name)
        if '/' in name:
            name = name.split('/')[-1]
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        return name

    def _select_intermediate_indices(self, sample_names: list) -> list:
        max_samples = int(getattr(self, 'eval_intermediate_max_samples', 0) or 0)
        if max_samples <= 0:
            return list(range(len(sample_names)))
        indices = []
        for i, name in enumerate(sample_names):
            key = self._parse_sample_name(name)
            if key in self._eval_intermediate_saved:
                continue
            indices.append(i)
            self._eval_intermediate_saved.add(key)
            if len(indices) >= max_samples:
                break
        return indices

    def _resolve_intermediate_steps(self, num_steps: int) -> list:
        steps = getattr(self, 'eval_intermediate_steps', None)
        if steps:
            return [int(s) for s in steps if 0 <= int(s) < num_steps]
        t_list = getattr(self, 'eval_intermediate_t', None)
        if not t_list:
            return []
        resolved = []
        for t in t_list:
            t_val = max(0.0, min(1.0, float(t)))
            idx = int(round(t_val * (num_steps - 1)))
            resolved.append(idx)
        return sorted(set(resolved))

    def _save_intermediate_images(
        self,
        sample_names: list,
        steps_dict: dict,
        out_dir: str,
        labels: list | None,
        final_img=None,
        include_final: bool = True,
    ) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ordered_steps = sorted(steps_dict.keys())
        print(f"[intermediate] _save_intermediate_images called: out_dir={out_dir}, sample_names={sample_names}, ordered_steps={ordered_steps}")
        if not ordered_steps:
            print(f"[intermediate][warn] steps_dict is empty! steps_dict={steps_dict}")
            return
        for i, name in enumerate(sample_names):
            sample_dir = out_path / self._parse_sample_name(name)
            sample_dir.mkdir(parents=True, exist_ok=True)
            row_imgs = []
            per_tile_labels = []
            for idx, step_idx in enumerate(ordered_steps):
                step_tensor = steps_dict.get(step_idx)
                if not torch.is_tensor(step_tensor):
                    continue
                img = step_tensor[i].detach().float().cpu()
                if img.dim() == 3:
                    img = img.squeeze(0)
                if img.min().item() < 0.0:
                    img = (img + 1.0) / 2.0
                img = img.clamp(0.0, 1.0)
                img = (img.numpy() * 255.0).astype(np.uint8)
                img = img.T
                row_imgs.append(img)
                # Save each step image (디버깅용 로그 추가)
                try:
                    print(f"[intermediate] Saving step image: {sample_dir / f'step_{step_idx}.png'}, shape={img.shape}")
                    step_img = Image.fromarray(img, mode="L")
                    step_img.save(sample_dir / f"step_{step_idx}.png")
                except Exception as e:
                    print(f"[intermediate][warn] Failed to save step image: {sample_dir / f'step_{step_idx}.png'}: {e}")
                if labels is not None and idx < len(labels):
                    per_tile_labels.append(labels[idx])
                else:
                    per_tile_labels.append(f"step={int(step_idx)}")
            if final_img is not None and include_final:
                img = final_img[i].detach().float().cpu()
                if img.dim() == 3:
                    img = img.squeeze(0)
                if img.min().item() < 0.0:
                    img = (img + 1.0) / 2.0
                img = img.clamp(0.0, 1.0)
                img = (img.numpy() * 255.0).astype(np.uint8)
                img = img.T
                row_imgs.append(img)
                per_tile_labels.append("t=1.00")
            if row_imgs:
                row = np.concatenate(row_imgs, axis=1)
                tile_h, tile_w = row_imgs[0].shape
                font_size = max(12, int(tile_h * 0.08))
                font = None
                for path in (
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                ):
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except Exception:
                        continue
                if font is None:
                    font = ImageFont.load_default()
                tmp = Image.new("L", (1, 1), color=255)
                draw_tmp = ImageDraw.Draw(tmp)
                text_heights = [draw_tmp.textbbox((0, 0), label, font=font)[3] for label in per_tile_labels]
                top_margin = (max(text_heights) if text_heights else font_size) + 6
                canvas = Image.new("L", (row.shape[1], row.shape[0] + top_margin), color=255)
                canvas.paste(Image.fromarray(row, mode="L"), (0, top_margin))
                draw = ImageDraw.Draw(canvas)
                for idx, label in enumerate(per_tile_labels):
                    x0 = idx * tile_w
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    x = x0 + (tile_w - text_w) // 2
                    y = 2
                    draw.text((x, y), label, fill=0, font=font)
                canvas.save(sample_dir / "steps_row.png")

    def _maybe_save_intermediate_eval(self, images: torch.Tensor, sample_names: list) -> None:
        print(f"[intermediate] _maybe_save_intermediate_eval called")
        if not getattr(self, 'eval_save_intermediate', False):
            print(f"[intermediate] eval_save_intermediate is False, skipping")
            return
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'is_global_zero') and not self.trainer.is_global_zero:
            print(f"[intermediate] Not global_zero, skipping")
            return
        out_dir = getattr(self, 'eval_intermediate_dir', None)
        if not out_dir:
            print(f"[intermediate] eval_intermediate_dir is None, skipping")
            return
        indices = self._select_intermediate_indices(sample_names)
        if not indices:
            print(f"[intermediate] No indices selected, skipping")
            return
        timesteps = int(getattr(self.hparams, 'timesteps', 0))
        if timesteps <= 0:
            print(f"[intermediate] timesteps <= 0, skipping")
            return
        save_steps = self._resolve_intermediate_steps(timesteps)
        if not save_steps:
            print(f"[intermediate] save_steps is empty, skipping")
            return
        label_list = None
        if self.eval_intermediate_t:
            label_list = []
            idx_map = {}
            for t in self.eval_intermediate_t:
                t_val = max(0.0, min(1.0, float(t)))
                idx = int(round(t_val * (timesteps - 1)))
                if idx not in idx_map:
                    idx_map[idx] = f"t={t_val:.2f}"
            for step_idx in sorted(save_steps):
                label_list.append(idx_map.get(step_idx, f"step={int(step_idx)}"))
        subset_images = images[indices]
        subset_names = [sample_names[i] for i in indices]
        
        if self.mode == 'dfm_binary':
            final = self.sample(None, subset_images)
        else:
            noise = self._sample_x0_like(subset_images, channels=self._infer_noise_channels())
            final = self.sample(noise, subset_images)
        
        # Save only final predictions
        if subset_images.shape[0] > 0:
            self._save_intermediate_images(
                subset_names,
                {},  # Empty dict - no intermediate steps
                out_dir,
                ["final"],  # Only final step
                final_img=final,
                include_final=True,
            )

    def training_step(self, batch, batch_idx):
        if self.mode == 'dfm_binary':
            return self._training_step_dfm(batch, batch_idx)

        images = batch['image']  # condition
        # geometry: soft label/distance map 지원 (XCA에서는 geometry 없으면 label 사용)
        geometry = batch.get('geometry', batch.get('label'))  # target
        labels = batch.get('label', geometry)
        
        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)

        if self.is_multitask:
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
            if geometry.dim() == 3:
                geometry = geometry.unsqueeze(1)
            labels_hard = labels * 2.0 - 1.0
            geometry_for_flow = torch.cat([labels_hard, geometry], dim=1)
            noise = self._sample_x0_like(geometry_for_flow)
            noise, geometry_for_flow, images, labels = random_patch_batch(
                [noise, geometry_for_flow, images, labels], patch_size, num_patches
            )
        else:
            # Prepare noise (x0) and target geometry (always Gaussian noise).
            noise = self._sample_x0_like(geometry)
            # Random patch extraction
            noise, geometry, images, labels = random_patch_batch(
                [noise, geometry, images, labels], patch_size, num_patches
            )
            geometry_for_flow = geometry
        
        # Flow matching: x (noise) -> geometry
        t, xt, ut = make_xt_continuous(self.flow_matcher, noise, geometry_for_flow)

        with torch.no_grad():
            # Diagnostics: noise vs target scale and early-t visualization.
            x1 = geometry_for_flow
            x0 = noise
            x1_norm = torch.linalg.vector_norm(x1.flatten(1), dim=1)
            x0_diff_norm = torch.linalg.vector_norm((x1 - x0).flatten(1), dim=1)
            ratio = (x0_diff_norm / (x1_norm + 1e-8)).mean()
            self.log('train/x1_x0_ratio', ratio, prog_bar=False, sync_dist=True)

            diag_every = getattr(self.hparams, 'log_diag_every_n_steps', 200)
            target_t = getattr(self.hparams, 'log_diag_t', 0.1)
            if (
                self.global_step % diag_every == 0
                and hasattr(self, 'logger')
                and self.logger is not None
                and hasattr(self.logger, 'experiment')
                and getattr(self.trainer, 'is_global_zero', True)
            ):
                idx = torch.argmin((t - target_t).abs()).item()
                xt_img = xt[idx, 0:1].detach()
                xt_min = xt_img.min()
                xt_max = xt_img.max()
                if (xt_max - xt_min) > 1e-6:
                    xt_img = (xt_img - xt_min) / (xt_max - xt_min)
                    self.logger.experiment.add_image(
                        f'train/xt_t{target_t:.2f}',
                        xt_img,
                        global_step=self.global_step,
                    )
        
        # UNet forward: (x=xt, time=t, cond=images)
        vlm_film_cond = self._get_vlm_film_cond(images, batch)
        unet_out = self._unet_forward(xt, t, images, batch=batch, vlm_cond=vlm_film_cond)
        if self.use_geometry_head:
            v = unet_out[:, 0:1, :, :]
            geometry_pred = unet_out[:, 1:2, :, :]
        else:
            v = unet_out
            geometry_pred = None

        if self.is_multitask:
            lambda_soft = float(getattr(self.hparams, 'lambda_soft', 1.0))
            v_hard, v_soft = v[:, 0:1, :, :], v[:, 1:2, :, :]
            ut_hard, ut_soft = ut[:, 0:1, :, :], ut[:, 1:2, :, :]
            xt_hard, xt_soft = xt[:, 0:1, :, :], xt[:, 1:2, :, :]
            geom_hard, geom_soft = geometry_for_flow[:, 0:1, :, :], geometry_for_flow[:, 1:2, :, :]

            if self.use_registry_loss:
                loss_hard, loss_dict_hard = self.loss_fn(
                    v_hard,
                    ut_hard,
                    xt_hard,
                    geom_hard,
                    t=t,
                    geometry_pred=geometry_pred,
                    hard_labels=labels,
                )
                loss_soft, loss_dict_soft = self.loss_fn(
                    v_soft,
                    ut_soft,
                    xt_soft,
                    geom_soft,
                    t=t,
                    geometry_pred=geometry_pred,
                    hard_labels=labels,
                )
                for name, value in loss_dict_hard.items():
                    self.log(f'train/hard_{name}_loss', value, prog_bar=False, sync_dist=True)
                for name, value in loss_dict_soft.items():
                    self.log(f'train/soft_{name}_loss', value, prog_bar=False, sync_dist=True)
            else:
                loss_hard = self._compute_loss(v_hard, ut_hard, xt_hard, geom_hard, t)
                loss_soft = self._compute_loss(v_soft, ut_soft, xt_soft, geom_soft, t)

            loss = loss_hard + lambda_soft * loss_soft
            self.log('train/loss_hard', loss_hard, prog_bar=False, sync_dist=True)
            self.log('train/loss_soft', loss_soft, prog_bar=False, sync_dist=True)
            self.log('train/lambda_soft', lambda_soft, prog_bar=False, sync_dist=True)
        else:
            # Compute loss based on loss_type
            loss, loss_dict = loss_cfm(
                use_registry_loss=self.use_registry_loss,
                loss_fn=self.loss_fn,
                compute_loss_fn=self._compute_loss,
                v=v,
                ut=ut,
                xt=xt,
                geometry=geometry_for_flow,
                t=t,
                geometry_pred=geometry_pred,
                hard_labels=labels,
            )
            for name, value in loss_dict.items():
                self.log(f'train/{name}_loss', value, prog_bar=False, sync_dist=True)

        # Log (sync_dist for DDP)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def _training_step_dfm(self, batch, batch_idx):
        """Training step for DFM binary mode."""
        images = batch['image']  # condition
        labels = batch.get('label', batch.get('geometry'))

        patch_size, num_patches = select_patch_params(self.hparams.patch_plan)

        # Binary FM uses hard labels only (keep dataset pipeline unchanged).
        if labels.dim() == 3:
            labels = labels.unsqueeze(1)
        x1 = to_binary_mask(labels)
        x1, images = random_patch_batch([x1, images], patch_size, num_patches)

        t, xt, x0 = make_xt_binary(x1)
        vlm_film_cond = self._get_vlm_film_cond(images, batch)
        
        # Pass gt_mask for junction gating if source='gt'
        gt_mask_for_gating = None
        if self.junction_gating_config is not None and self.junction_gating_config.get('enabled', False):
            if self.junction_gating_config.get('source') == 'gt':
                gt_mask_for_gating = x1
        
        unet_out = self._unet_forward(xt, t, images, batch=batch, vlm_cond=vlm_film_cond, gt_mask=gt_mask_for_gating)
        logits = unet_out[:, 1:2, :, :] if self.use_geometry_head else unet_out

        loss, loss_dict = loss_dfm_binary(logits, x1)

        # Self-consistency loss (velocity-based): enforce straight flows.
        if torch.rand((), device=x1.device).item() < 0.2:
            batch_size = x1.shape[0]
            device = x1.device
            reg_bs = 2 if batch_size > 2 else batch_size
            if reg_bs < batch_size:
                idx = torch.randperm(batch_size, device=device)[:reg_bs]
                x1_reg = x1.index_select(0, idx)
                x0_reg = x0.index_select(0, idx)
                images_reg = images.index_select(0, idx)
                batch_reg = self._subset_batch_for_vlm(batch, idx)
            else:
                x1_reg = x1
                x0_reg = x0
                images_reg = images
                batch_reg = batch

            t1 = torch.rand(reg_bs, device=device) * 0.4
            t2 = t1 + 0.1 + torch.rand(reg_bs, device=device) * 0.4
            t_mid = (t1 + t2) / 2.0
            dt1 = t_mid - t1
            dt2 = t2 - t_mid

            mask1 = torch.rand_like(x1_reg) < t1.view(-1, 1, 1, 1)
            x_t1 = torch.where(mask1, x1_reg, x0_reg)
            p_t1 = x_t1.float()

            # Velocity at t1
            logits_t1 = self._unet_forward(x_t1, t1, images_reg, batch=batch_reg)
            p1_t1 = torch.sigmoid(logits_t1)
            v_t1 = (p1_t1 - p_t1) / (1.0 - t1.view(-1, 1, 1, 1) + 1e-5)

            # Euler step: t1 -> t_mid
            p_mid = torch.clamp(p_t1 + v_t1 * dt1.view(-1, 1, 1, 1), 0.0, 1.0)

            # Velocity at t_mid
            logits_mid = self._unet_forward(p_mid, t_mid, images_reg, batch=batch_reg)
            p1_mid = torch.sigmoid(logits_mid)
            v_mid = (p1_mid - p_mid) / (1.0 - t_mid.view(-1, 1, 1, 1) + 1e-5)

            # Two-step result: t_mid -> t2
            p_t2_two = p_mid + v_mid * dt2.view(-1, 1, 1, 1)

            # Direct: t1 -> t2
            p_t2_direct = p_t1 + v_t1 * (t2 - t1).view(-1, 1, 1, 1)

            consistency_loss = F.mse_loss(p_t2_direct, p_t2_two)
            loss = loss + 0.1 * consistency_loss
            self.log('train/consistency_loss', consistency_loss, prog_bar=False, sync_dist=True)
        self.log('train/dfm_bce_loss', loss_dict['bce'], prog_bar=False, sync_dist=True)
        self.log('train/dfm_dice_loss', loss_dict['dice'], prog_bar=False, sync_dist=True)

        if getattr(self.hparams, 'debug_dfm', False):
            assert logits.shape == x1.shape, "DFM logits/x1 shape mismatch"
            assert torch.isfinite(loss).all().item(), "DFM loss is not finite"
            xt_min = float(xt.min().item())
            xt_max = float(xt.max().item())
            assert 0.0 <= xt_min <= 1.0 and 0.0 <= xt_max <= 1.0, "DFM xt out of [0,1]"
            assert ((xt == 0) | (xt == 1)).all().item(), "DFM xt is not binary"
            t_small = t < 0.05
            if t_small.any():
                match_rate = (xt[t_small] == x0[t_small]).float().mean().item()
                assert match_rate > 0.6, "DFM xt not close to x0 for small t"
            t_large = t > 0.95
            if t_large.any():
                match_rate = (xt[t_large] == x1[t_large]).float().mean().item()
                assert match_rate > 0.6, "DFM xt not close to x1 for large t"

        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def _compute_loss(self, v, ut, xt, geometry, t):
        """Compute loss based on loss_type configuration."""
        losses = {}

        # Base flow matching loss:
        # Default is L2(MSE) to match standard flow-matching objective.
        # For backward compatibility, legacy 'l1*' loss_type keeps L1 base.
        use_l1_base = isinstance(self.loss_type, str) and self.loss_type.startswith('l1')
        if use_l1_base:
            base_loss = torch.abs(v - ut).mean()
            losses['l1'] = base_loss
        else:
            base_loss = ((v - ut) ** 2).mean()
            losses['l2'] = base_loss
        
        # Additional losses based on loss_type
        if self.loss_type in {'l1', 'l2'}:
            # Default: base loss only
            total_loss = base_loss
        
        elif self.loss_type in {'l1_bce', 'l2_bce'}:
            # Base + BCE (recommended for SAUNA soft labels)
            # Compute output geometry from xt for BCE loss
            # Approximate output: xt at t=1 should be close to geometry
            # Use xt as proxy for output geometry (or compute from flow)
            output_geometry = xt  # Simplified: use current state as proxy
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry
            
            # Convert to [0, 1] range if needed (SAUNA is already in [0, 1])
            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            
            # BCE loss on probabilities
            eps = 1e-7
            output_probs = torch.clamp(output_probs, eps, 1 - eps)
            bce_loss = -(target_probs * torch.log(output_probs) + 
                        (1 - target_probs) * torch.log(1 - output_probs)).mean()
            losses['bce'] = bce_loss
            
            total_loss = base_loss + self.bce_weight * bce_loss
        
        elif self.loss_type in {'l1_l2', 'l2_l2'}:
            # Base + additional L2 regularizer on (v - ut)
            # (Kept for backward compatibility/tuning.)
            l2_loss = ((v - ut) ** 2).mean()
            losses['l2'] = l2_loss
            total_loss = base_loss + self.l2_weight * l2_loss
        
        elif self.loss_type in {'l1_bce_l2', 'l2_bce_l2'}:
            # Base + BCE + additional L2 regularizer
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
            l2_loss = ((v - ut) ** 2).mean()
            
            losses['bce'] = bce_loss
            losses['l2'] = l2_loss
            total_loss = base_loss + self.bce_weight * bce_loss + self.l2_weight * l2_loss
        
        elif self.loss_type in {'l1_bce_dice', 'l2_bce_dice'}:
            # Base + BCE + Dice
            output_geometry = xt
            if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
                output_geometry = output_geometry.squeeze(1)
            if geometry.dim() == 4 and geometry.shape[1] == 1:
                geometry_2d = geometry.squeeze(1)
            else:
                geometry_2d = geometry
            
            output_probs = torch.clamp(output_geometry, 0.0, 1.0)
            target_probs = torch.clamp(geometry_2d, 0.0, 1.0)
            
            # BCE loss
            eps = 1e-7
            output_probs_clamped = torch.clamp(output_probs, eps, 1 - eps)
            bce_loss = -(target_probs * torch.log(output_probs_clamped) + 
                        (1 - target_probs) * torch.log(1 - output_probs_clamped)).mean()
            
            # Dice loss
            dice_loss = self.dice_loss(output_probs.unsqueeze(1), target_probs)
            
            losses['bce'] = bce_loss
            losses['dice'] = dice_loss
            total_loss = base_loss + self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        else:
            # Unknown loss_type, fallback to base loss
            total_loss = base_loss
        
        # Log individual losses
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}_loss', loss_value, prog_bar=False, sync_dist=True)
        
        return total_loss

    def make_x0_mixed(self, x1: torch.Tensor, stage: str, global_step: int) -> torch.Tensor:
        """Generate x0 by mixing near-start (blurred x1) and noise-start."""
        p_start = float(getattr(self.hparams, 'x0_p_start', 0.8))
        p_end = float(getattr(self.hparams, 'x0_p_end', 0.0))
        decay_steps = float(getattr(self.hparams, 'x0_p_decay_steps', 50000))
        x0_alpha = float(getattr(self.hparams, 'x0_alpha', 0.1))
        x0_sigma = float(getattr(self.hparams, 'x0_sigma', 0.1))
        blur_sigma = float(getattr(self.hparams, 'x0_blur_sigma', 4.0))

        if decay_steps <= 0:
            p = p_end
        else:
            progress = min(max(global_step / decay_steps, 0.0), 1.0)
            p = p_end + (p_start - p_end) * (1.0 - progress)

        if stage in {'val', 'test'} and getattr(self.hparams, 'debug_val_use_near_start', False):
            p = 1.0

        # Near-start: blur x1 then add small noise.
        if blur_sigma > 0:
            radius = int(torch.ceil(torch.tensor(3.0 * blur_sigma)).item())
            kernel_size = radius * 2 + 1
            device = x1.device
            dtype = x1.dtype
            coords = torch.arange(kernel_size, device=device, dtype=dtype) - radius
            kernel_1d = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
            kernel_y = kernel_1d.view(1, 1, kernel_size, 1)
            channels = x1.shape[1]
            kernel_x = kernel_x.repeat(channels, 1, 1, 1)
            kernel_y = kernel_y.repeat(channels, 1, 1, 1)
            padding = (radius, radius, radius, radius)
            blurred = F.pad(x1, padding, mode='reflect')
            blurred = F.conv2d(blurred, kernel_x, groups=channels)
            blurred = F.conv2d(blurred, kernel_y, groups=channels)
        else:
            blurred = x1
        near = blurred + x0_alpha * torch.randn_like(x1)

        # Noise-start: pure Gaussian noise.
        noise = x0_sigma * torch.randn_like(x1)

        # Sample mixing mask per sample.
        if p <= 0:
            x0 = noise
            near_rate = 0.0
        elif p >= 1:
            x0 = near
            near_rate = 1.0
        else:
            mask = (torch.rand(x1.shape[0], device=x1.device) < p).float().view(-1, 1, 1, 1)
            x0 = mask * near + (1 - mask) * noise
            near_rate = mask.mean().item()

        # Keep x0 in the same [-1, 1] range as geometry.
        x0 = torch.clamp(x0, -1.0, 1.0)

        if getattr(self.hparams, 'use_x0_mixing', False):
            self.log(f'{stage}/x0_near_rate', near_rate, prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x0_mean', x0.mean(), prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x0_std', x0.std(), prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x1_mean', x1.mean(), prog_bar=False, sync_dist=True)
            self.log(f'{stage}/x1_std', x1.std(), prog_bar=False, sync_dist=True)

        return x0

    def _sample_x0_like(self, ref: torch.Tensor, channels: int | None = None) -> torch.Tensor:
        """Sample x0 from N(0, 1)."""
        if channels is None:
            return torch.randn_like(ref)
        shape = list(ref.shape)
        shape[1] = channels
        return torch.randn(shape, device=ref.device, dtype=ref.dtype)


    def sample(self, noise, images):
        """Sample from flow matching model (inference).
        
        Args:
            noise: (B, 1, H, W) - initial noise (ignored for dfm_binary)
            images: (B, 1, H, W) - condition images
        
        Returns:
            output_geometry (B, 1, H, W)
        """
        if self.mode == 'dfm_binary':
            sampler = sampler_dfm_euler
            if getattr(self.hparams, 'dfm_sampler', self.dfm_sampler) == 'heun':
                sampler = sampler_dfm_heun
            eps = float(getattr(self.hparams, 'dfm_eps', 1e-6))
            return sampler(
                self._dfm_logits,
                images,
                self.hparams.timesteps,
                eps=eps,
            )
        
        # Store images for ode_func
        self._sample_images = images
        
        traj = odeint(
            self.ode_func,
            noise,
            torch.linspace(0, 1, self.hparams.timesteps, device=noise.device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5"
        )
        
        return traj[-1]

    def _dfm_logits(self, xt: torch.Tensor, t: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Forward helper for DFM logits (single-head or geometry head).
        
        Note: For VLM-FiLM, we extract VLM features from the current images.
        In sliding window inference, each patch extracts VLM features independently,
        matching the training behavior.
        """
        # Extract VLM conditioning if enabled (same as training)
        vlm_cond = self._get_vlm_film_cond(images, batch=None) if self.use_vlm_film else None
        unet_out = self._unet_forward(xt, t, images, batch=None, vlm_cond=vlm_cond)
        if self.use_geometry_head:
            return unet_out[:, 1:2, :, :]
        return unet_out

    def validation_step(self, batch, batch_idx):
        if self.mode == 'dfm_binary':
            return self._validation_step_dfm(batch, batch_idx)

        images = batch['image']  # condition
        labels = batch['label']
        geometry = batch.get('geometry', batch.get('label'))  # target
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        # intermediate saving is handled in test only
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                if getattr(self.hparams, 'use_sliding_infer', True):
                    output_geometry = self._infer_sliding(images)
                else:
                    output_geometry = self._infer_full(images)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            if getattr(self.hparams, 'use_sliding_infer', True):
                output_geometry = self._infer_sliding(images)
            else:
                output_geometry = self._infer_full(images)
        
        output_geometry_hard = output_geometry
        output_geometry_soft = None
        if self.is_multitask and output_geometry.dim() == 4 and output_geometry.shape[1] >= 2:
            output_geometry_hard = output_geometry[:, 0:1, :, :]
            output_geometry_soft = output_geometry[:, 1:2, :, :]

        # Compute reconstruction loss (final generation quality)
        loss_pred = output_geometry_soft if output_geometry_soft is not None else output_geometry_hard
        loss = torch.abs(loss_pred - geometry).mean()
        self.log('val/reconstruction_loss', loss, prog_bar=True, sync_dist=True)
        
        # Convert predictions to class indices
        if output_geometry_hard.dim() == 4 and output_geometry_hard.shape[1] == 1:
            output_geometry_hard = output_geometry_hard.squeeze(1)
        # Debug stats for output distribution (helps diagnose noisy preds).
        out_min = output_geometry_hard.min()
        out_max = output_geometry_hard.max()
        out_mean = output_geometry_hard.mean()
        out_std = output_geometry_hard.std()
        self.log('val/output_geometry_min', out_min, prog_bar=False, sync_dist=True)
        self.log('val/output_geometry_max', out_max, prog_bar=False, sync_dist=True)
        self.log('val/output_geometry_mean', out_mean, prog_bar=False, sync_dist=True)
        self.log('val/output_geometry_std', out_std, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_min', out_min, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_max', out_max, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_mean', out_mean, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_std', out_std, prog_bar=False, sync_dist=True)
        
        # Threshold on logits (0 기준)
        preds = (output_geometry_hard > 0.0).long()
        if batch_idx == 0 and not getattr(self.trainer, "sanity_checking", False):
            from pytorch_lightning.utilities.rank_zero import rank_zero_info
            rank_zero_info(
                "val/x1_pred stats: "
                f"min={out_min.item():.4f} max={out_max.item():.4f} "
                f"mean={out_mean.item():.4f} std={out_std.item():.4f}"
            )
        
        # Convert geometry for logging (ensure same dimensions as output_geometry)
        if geometry.dim() == 4 and geometry.shape[1] == 1:
            geometry = geometry.squeeze(1)
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log (sync_dist for DDP)
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True, sync_dist=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False, sync_dist=True)

        # Sanity check 단계에서는 이미지 로깅을 스킵해 불필요한 all_gather_object 오버헤드 제거
        if getattr(self.trainer, "sanity_checking", False):
            return general_metrics['dice']
        
        # Queue images for logging (handles DDP sharding safely).
        self._queue_images_for_logging(
            sample_names=sample_names,
            images=images,
            labels=labels,
            preds=preds,
            tag_prefix='val',
            geometry=geometry,
            output_geometry=output_geometry_hard,
        )
        
        return general_metrics['dice']

    def _validation_step_dfm(self, batch, batch_idx):
        """Validation step for DFM binary mode."""
        images = batch['image']  # condition
        labels = batch['label']
        geometry = labels.float()
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        # intermediate saving is handled in test only

        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                if getattr(self.hparams, 'use_sliding_infer', True):
                    output_geometry = self._sliding_window_predict(images)
                else:
                    output_geometry = self.sample(None, images)
                output_geometry_list.append(output_geometry)
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            if getattr(self.hparams, 'use_sliding_infer', True):
                output_geometry = self._sliding_window_predict(images)
            else:
                output_geometry = self.sample(None, images)

        output_geometry_hard = output_geometry

        # Compute BCE + Dice on probabilities for DFM validation loss
        output_probs = torch.clamp(output_geometry_hard, 0.0, 1.0)
        target_probs = torch.clamp(geometry, 0.0, 1.0)
        if output_probs.dim() == 3:
            output_probs = output_probs.unsqueeze(1)
        if target_probs.dim() == 3:
            target_probs = target_probs.unsqueeze(1)
        # AMP-safe BCE (disable autocast for numeric stability)
        with torch.autocast(device_type="cuda", enabled=False):
            bce_loss = F.binary_cross_entropy(output_probs.float(), target_probs.float())
            dice_loss = _soft_dice_loss(output_probs.float(), target_probs.float())
        loss = bce_loss + dice_loss
        self.log('val/dfm_bce_loss', bce_loss, prog_bar=False, sync_dist=True)
        self.log('val/dfm_dice_loss', dice_loss, prog_bar=False, sync_dist=True)
        self.log('val/reconstruction_loss', loss, prog_bar=True, sync_dist=True)

        # Convert predictions to class indices
        if output_geometry_hard.dim() == 4 and output_geometry_hard.shape[1] == 1:
            output_geometry_hard = output_geometry_hard.squeeze(1)

        out_min = output_geometry_hard.min()
        out_max = output_geometry_hard.max()
        out_mean = output_geometry_hard.mean()
        out_std = output_geometry_hard.std()
        self.log('val/output_geometry_min', out_min, prog_bar=False, sync_dist=True)
        self.log('val/output_geometry_max', out_max, prog_bar=False, sync_dist=True)
        self.log('val/output_geometry_mean', out_mean, prog_bar=False, sync_dist=True)
        self.log('val/output_geometry_std', out_std, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_min', out_min, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_max', out_max, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_mean', out_mean, prog_bar=False, sync_dist=True)
        self.log('val/x1_pred_std', out_std, prog_bar=False, sync_dist=True)

        # Threshold on probabilities (0.5 기준)
        preds = (output_geometry_hard > 0.5).long()

        # Convert geometry for logging
        if geometry.dim() == 4 and geometry.shape[1] == 1:
            geometry = geometry.squeeze(1)

        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)

        # Log (sync_dist for DDP)
        self.log_dict({'val/' + k: v for k, v in general_metrics.items()}, prog_bar=True, sync_dist=True)
        self.log_dict({'val/' + k: v for k, v in vessel_metrics.items()}, prog_bar=False, sync_dist=True)

        if getattr(self.trainer, "sanity_checking", False):
            return general_metrics['dice']

        # Queue images for logging (handles DDP sharding safely).
        self._queue_images_for_logging(
            sample_names=sample_names,
            images=images,
            labels=labels,
            preds=preds,
            tag_prefix='val',
            geometry=geometry,
            output_geometry=output_geometry_hard,
        )

        return general_metrics['dice']

    def on_validation_epoch_end(self):
        self._flush_queued_images()

    def test_step(self, batch, batch_idx):
        if self.mode == 'dfm_binary':
            return self._test_step_dfm(batch, batch_idx)

        images = batch['image']  # condition
        labels = batch['label']
        
        # Get sample names if available
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])
        
        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        self._maybe_save_intermediate_eval(images, sample_names)
        
        # Ensemble: multiple sampling and averaging
        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                if getattr(self.hparams, 'use_sliding_infer', True):
                    output_geometry = self._infer_sliding(images)
                else:
                    output_geometry = self._infer_full(images)
                output_geometry_list.append(output_geometry)
            # Average predictions
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            # Single sampling
            if getattr(self.hparams, 'use_sliding_infer', True):
                output_geometry = self._infer_sliding(images)
            else:
                output_geometry = self._infer_full(images)
        
        output_geometry_hard = output_geometry
        if self.is_multitask and output_geometry.dim() == 4 and output_geometry.shape[1] >= 2:
            output_geometry_hard = output_geometry[:, 0:1, :, :]

        # Convert predictions to class indices
        if output_geometry_hard.dim() == 4 and output_geometry_hard.shape[1] == 1:
            output_geometry_hard = output_geometry_hard.squeeze(1)
        
        # Threshold on logits (0 기준)
        preds = (output_geometry_hard > 0.0).long()
        
        # Compute metrics
        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)
        
        # Log (sync_dist for DDP)
        self.log_dict({'test/' + k: v for k, v in general_metrics.items()}, sync_dist=True)
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()}, sync_dist=True)
        
        self._log_images(sample_names, images, labels, preds, tag_prefix='test')
        
        # Store predictions for logging
        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks_binary = (preds > 0).float()
            label_masks = (labels > 0).float()

    def _test_step_dfm(self, batch, batch_idx):
        """Test step for DFM binary mode."""
        images = batch['image']  # condition
        labels = batch['label']
        sample_names = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(images.shape[0])])

        # Convert labels for metrics
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = (labels > 0.5).long()

        self._maybe_save_intermediate_eval(images, sample_names)

        if self.hparams.num_ensemble > 1:
            output_geometry_list = []
            for _ in range(self.hparams.num_ensemble):
                if getattr(self.hparams, 'use_sliding_infer', True):
                    output_geometry = self._sliding_window_predict(images)
                else:
                    output_geometry = self.sample(None, images)
                output_geometry_list.append(output_geometry)
            output_geometry = torch.stack(output_geometry_list).mean(dim=0)
        else:
            if getattr(self.hparams, 'use_sliding_infer', True):
                output_geometry = self._sliding_window_predict(images)
            else:
                output_geometry = self.sample(None, images)

        if output_geometry.dim() == 4 and output_geometry.shape[1] == 1:
            output_geometry = output_geometry.squeeze(1)

        preds = (output_geometry > 0.5).long()

        general_metrics = self.val_metrics(preds, labels)
        vessel_metrics = self.vessel_metrics(preds, labels)

        self.log_dict({'test/' + k: v for k, v in general_metrics.items()}, sync_dist=True)
        self.log_dict({'test/' + k: v for k, v in vessel_metrics.items()}, sync_dist=True)

        self._log_images(sample_names, images, labels, preds, tag_prefix='test')

        if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'save_predictions'):
            pred_masks_binary = (preds > 0).float()
            label_masks = (labels > 0).float()
            
            # Prepare metrics for each sample
            sample_metrics = []
            for i in range(images.shape[0]):
                sample_metric = {}
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in general_metrics.items()})
                sample_metric.update({k: v.item() if torch.is_tensor(v) else v for k, v in vessel_metrics.items()})
                sample_metrics.append(sample_metric)
            
            # Save predictions
            self.trainer.logger.save_predictions(
                sample_names, images, pred_masks_binary, label_masks, sample_metrics
            )
        
        return general_metrics['dice']

    @torch.no_grad()
    def ode_func(self, t, x):
        """ODE function for flow matching.
        
        Args:
            t: Time step (scalar or tensor)
            x: State tensor (B, 1, H, W) - noise/geometry at time t
        
        Returns:
            Velocity field from UNet
        """
        if isinstance(t, torch.Tensor):
            t = t.expand(x.shape[0])
        else:
            t = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        
        # Get condition images (stored during sample)
        images = self._sample_images
        
        # UNet forward: (x=noise, time=t, cond=images)
        unet_out = self._unet_forward(x, t, images, batch=None)
        if self.use_geometry_head:
            return unet_out[:, 0:1, :, :]
        return unet_out

    def on_save_checkpoint(self, checkpoint):
        """Save metadata about which stages FiLM was applied to during training."""
        super().on_save_checkpoint(checkpoint)
        if self.use_vlm_film and self._vlm_film_stage_indices is not None:
            checkpoint['vlm_film_trained_stages'] = self._vlm_film_stage_indices

    def on_load_checkpoint(self, checkpoint):
        """Hook called after loading checkpoint to sync _vlm_film_stage_indices with actual trained stages."""
        super().on_load_checkpoint(checkpoint)
        
        # Priority 1: Use trained stages from checkpoint metadata (most reliable)
        trained_stages = checkpoint.get('vlm_film_trained_stages')
        if trained_stages is not None:
            self._vlm_film_stage_indices = sorted(trained_stages)
            print(f"[on_load_checkpoint] Using trained stages from checkpoint: {self._vlm_film_stage_indices}")
            return
        
        # Priority 2: Use hparams if available
        if self.use_vlm_film and hasattr(self.hparams, 'vlm_film_decoder_stages'):
            decoder_stages = self.hparams.vlm_film_decoder_stages
            if decoder_stages is not None:
                self._vlm_film_stage_indices = sorted(decoder_stages)
                print(f"[on_load_checkpoint] Using stages from hparams: {self._vlm_film_stage_indices}")
                return
        
        # Priority 3: Legacy checkpoint - infer stages from FiLM head output dimensions
        # Decoder stage channels: Stage0=128, Stage1=96, Stage2=64, Stage3=32
        # FiLM outputs gamma+beta: Stage0=256, Stage1=192, Stage2=128, Stage3=64
        if self.use_vlm_film and hasattr(self, 'vlm_film_heads') and self.vlm_film_heads is not None:
            dim_to_stage = {256: 0, 192: 1, 128: 2, 64: 3}
            inferred_stages = []
            for head_idx, head in enumerate(self.vlm_film_heads):
                # Get output dimension from final layer
                if hasattr(head, 'mlp') and len(head.mlp) > 2:
                    out_dim = head.mlp[2].out_features
                    stage = dim_to_stage.get(out_dim)
                    if stage is not None:
                        inferred_stages.append(stage)
            
            if inferred_stages:
                self._vlm_film_stage_indices = sorted(inferred_stages)
                print(f"[on_load_checkpoint] Inferred stages from head output dims: {self._vlm_film_stage_indices}")
                return
            else:
                # Fallback to old behavior if dimension mapping fails
                num_heads = len(self.vlm_film_heads)
                inferred_stages = list(range(num_heads))
                self._vlm_film_stage_indices = inferred_stages
                print(f"[on_load_checkpoint] Fallback: Inferred stages from {num_heads} heads: {inferred_stages}")
                return
        
        print(f"[on_load_checkpoint] WARNING: Could not determine FiLM stages!")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # val/dice는 높을수록 좋음
            patience=5,  # validation 주기(25 epoch)를 고려 = 75 epoch
            factor=0.5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/dice',  # ✅ 올바른 metric
                'interval': 'epoch',
                'frequency': 15,  # validation 주기와 일치
            }
        }



    def _log_images(
        self,
        sample_names,
        images,
        labels,
        preds,
        tag_prefix: str,
        geometry=None,
        output_geometry=None,
        **_,
    ):
        """Log a batch of images to TensorBoard (single-process helper)."""
        # Check if logging is enabled
        if not hasattr(self.hparams, 'log_image_enabled') or not self.hparams.log_image_enabled:
            return
        
        # DDP: only log on rank 0
        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return
        
        log_names = getattr(self.hparams, 'log_image_names', None)
        
        for i, name in enumerate(sample_names):
            filename = name.split('/')[-1] if '/' in name else name
            if log_names is not None and filename not in log_names:
                continue

            self._log_one_image(
                filename=filename,
                image=images[i],
                label=labels[i],
                pred=preds[i],
                tag_prefix=tag_prefix,
                geometry=(geometry[i] if geometry is not None else None),
                output_geometry=(output_geometry[i] if output_geometry is not None else None),
            )

    def _log_one_image(
        self,
        *,
        filename: str,
        image: torch.Tensor,
        label: torch.Tensor,
        pred: torch.Tensor,
        tag_prefix: str,
        geometry: torch.Tensor | None = None,
        output_geometry: torch.Tensor | None = None,
    ) -> None:
        """Log a single sample with separate panels (readable in TensorBoard)."""
        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return

        def ensure_chw(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                return x.unsqueeze(0)
            return x

        img = ensure_chw((image + 1) / 2).clamp(0.0, 1.0)
        lab = ensure_chw(label.float()).clamp(0.0, 1.0)
        prd = ensure_chw(pred.float()).clamp(0.0, 1.0)

        base = f"{tag_prefix}/{filename}"
        self.logger.experiment.add_image(f"{base}/image", img, self.global_step)
        self.logger.experiment.add_image(f"{base}/label", lab, self.global_step)
        if geometry is not None:
            geo = ensure_chw(geometry.float()).clamp(0.0, 1.0)
            self.logger.experiment.add_image(f"{base}/geometry", geo, self.global_step)
        if output_geometry is not None:
            out_geo = ensure_chw(output_geometry.float()).clamp(0.0, 1.0)
            self.logger.experiment.add_image(f"{base}/output_geometry", out_geo, self.global_step)
        self.logger.experiment.add_image(f"{base}/pred", prd, self.global_step)

    def _queue_images_for_logging(
        self,
        *,
        sample_names,
        images: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        tag_prefix: str,
        geometry: torch.Tensor | None = None,
        output_geometry: torch.Tensor | None = None,
    ) -> None:
        """Queue selected images for logging; safe under DDP sharding."""
        if not hasattr(self.hparams, 'log_image_enabled') or not self.hparams.log_image_enabled:
            return

        log_names = getattr(self.hparams, 'log_image_names', None)
        if log_names is None:
            return

        for i, name in enumerate(sample_names):
            filename = name.split('/')[-1] if '/' in name else name
            if filename not in log_names:
                continue

            self._pending_image_logs.append(
                {
                    'tag_prefix': tag_prefix,
                    'filename': filename,
                    'image': images[i].detach().cpu(),
                    'label': labels[i].detach().cpu(),
                    'pred': preds[i].detach().cpu(),
                    'geometry': (geometry[i].detach().cpu() if geometry is not None else None),
                    'output_geometry': (output_geometry[i].detach().cpu() if output_geometry is not None else None),
                }
            )

    def _flush_queued_images(self) -> None:
        """Gather queued images across ranks and log them once on rank 0."""
        if not self._pending_image_logs:
            return

        gathered: list[list[dict]] | None = None
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, self._pending_image_logs)
        else:
            gathered = [self._pending_image_logs]

        # Clear local buffer ASAP to avoid growth if something goes wrong later.
        self._pending_image_logs = []

        # Only log on rank 0 (logger exists only there in this runner).
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        if not hasattr(self, 'logger') or self.logger is None or not hasattr(self.logger, 'experiment'):
            return

        # Flatten and de-duplicate by filename (DDP may still duplicate in some setups).
        flat: list[dict] = []
        for part in gathered:
            if part:
                flat.extend(part)

        seen = set()
        for item in flat:
            key = (item.get('tag_prefix'), item.get('filename'))
            if key in seen:
                continue
            seen.add(key)
            self._log_one_image(
                filename=item['filename'],
                image=item['image'],
                label=item['label'],
                pred=item['pred'],
                tag_prefix=item['tag_prefix'],
                geometry=item.get('geometry'),
                output_geometry=item.get('output_geometry'),
            )
