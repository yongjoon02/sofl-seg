"""Compatibility shim for VLM conditioning (moved to src.archs.components.vlm_conditioner)."""

from src.archs.components.vlm_conditioner import (
    VLMConditioner,
    QwenVLMTextGenerator,
    TextToConditionVector,
    AdaptiveFiLMHead,
)

# Backward compatibility
VLMFiLMConditioner = VLMConditioner

__all__ = [
    "VLMConditioner",
    "VLMFiLMConditioner",
    "QwenVLMTextGenerator",
    "TextToConditionVector",
    "AdaptiveFiLMHead",
]
