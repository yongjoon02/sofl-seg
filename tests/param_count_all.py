"""Count parameters for diffusion + flow models from config files."""

from __future__ import annotations
import autorootcwd
from pathlib import Path
import importlib.util
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIFFUSION_UNET_PATH = ROOT / "src" / "archs" / "components" / "diffusion_unet.py"


def _load_diffusion_unet_module():
    spec = importlib.util.spec_from_file_location("diffusion_unet", DIFFUSION_UNET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_self_attn_flags() -> tuple[bool, ...]:
    return (False, False, True, True)


def _as_tuple(value, default):
    if value is None:
        return default
    return tuple(value)


def _build_segdiff(cfg: dict, unet_module) -> object:
    return unet_module.SegDiffUNet(
        dim=cfg.get("dim", 64),
        image_size=cfg.get("image_size", 224),
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=_full_self_attn_flags(),
        rrdb_blocks=cfg.get("rrdb_blocks", 3),
    )


def _build_medsegdiff(cfg: dict, unet_module) -> object:
    return unet_module.MedSegDiffUNet(
        dim=cfg.get("dim", 64),
        image_size=cfg.get("image_size", 224),
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=_full_self_attn_flags(),
        mid_transformer_depth=cfg.get("mid_transformer_depth", 0),
        conditioning_fmap_size=cfg.get("conditioning_fmap_size"),
    )


def _build_colddiff(cfg: dict, unet_module) -> object:
    return unet_module.MedSegDiffUNet(
        dim=cfg.get("dim", 64),
        image_size=cfg.get("image_size", 224),
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=_full_self_attn_flags(),
        mid_transformer_depth=cfg.get("mid_transformer_depth", 1),
        conditioning_fmap_size=cfg.get("conditioning_fmap_size"),
    )


def _build_berdiff(cfg: dict, unet_module) -> object:
    return unet_module.SimpleConcatUNet(
        dim=cfg.get("dim", 64),
        image_size=cfg.get("image_size", 224),
        mask_channels=1,
        input_img_channels=1,
        dim_mult=(1, 2, 4, 8),
        full_self_attn=_full_self_attn_flags(),
    )


def _attn_flags_from_resolutions(image_size: int, channel_mult: tuple[int, ...], attn_resolutions: tuple[int, ...]):
    attn_set = set(attn_resolutions)
    flags = []
    curr = image_size
    for _ in channel_mult:
        flags.append(curr in attn_set)
        curr = max(1, curr // 2)
    return tuple(flags)


def _build_segdiff_flow(cfg: dict, unet_module) -> object:
    image_size = cfg.get("image_size", 224)
    channel_mult = _as_tuple(cfg.get("channel_mult"), (1, 2, 4, 8))
    attn_resolutions = _as_tuple(cfg.get("attn_resolutions"), (32, 16, 8, 8))
    full_self_attn = _attn_flags_from_resolutions(image_size, channel_mult, attn_resolutions)
    return unet_module.SegDiffUNet(
        dim=cfg.get("model_channels", cfg.get("dim", 64)),
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=channel_mult,
        full_self_attn=full_self_attn,
        rrdb_blocks=cfg.get("rrdb_blocks", 3),
    )


def _build_medsegdiff_flow(cfg: dict, unet_module) -> object:
    image_size = cfg.get("image_size", 224)
    channel_mult = _as_tuple(cfg.get("channel_mult"), (1, 2, 4, 8))
    attn_resolutions = _as_tuple(cfg.get("attn_resolutions"), (32, 16, 8, 8))
    full_self_attn = _attn_flags_from_resolutions(image_size, channel_mult, attn_resolutions)
    return unet_module.MedSegDiffUNet(
        dim=cfg.get("model_channels", cfg.get("dim", 64)),
        image_size=image_size,
        mask_channels=1,
        input_img_channels=1,
        dim_mult=channel_mult,
        full_self_attn=full_self_attn,
        mid_transformer_depth=cfg.get("mid_transformer_depth", 1),
    )


ARCH_BUILDERS = {
    "segdiff": _build_segdiff,
    "medsegdiff": _build_medsegdiff,
    "colddiff": _build_colddiff,
    "berdiff": _build_berdiff,
    "segdiff_flow": _build_segdiff_flow,
    "medsegdiff_flow": _build_medsegdiff_flow,
}


def _iter_config_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.yaml"))


def main() -> None:
    unet_module = _load_diffusion_unet_module()
    config_roots = [ROOT / "configs" / "diffusion", ROOT / "configs" / "flow"]
    config_paths: list[Path] = []
    for root in config_roots:
        if root.exists():
            config_paths.extend(_iter_config_paths(root))

    if not config_paths:
        print("No config files found.")
        return

    for path in config_paths:
        data = yaml.safe_load(path.read_text()) or {}
        model_cfg = data.get("model", {})
        arch_name = model_cfg.get("arch_name")
        if arch_name not in ARCH_BUILDERS:
            continue
        model = ARCH_BUILDERS[arch_name](model_cfg, unet_module)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{path.relative_to(ROOT)} | {arch_name} | {total_params:,}")


if __name__ == "__main__":
    main()
