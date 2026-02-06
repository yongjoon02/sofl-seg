"""
VLM-to-FiLM Conditioning for Flow-Matching Segmentation.

이 패키지는 Qwen2.5-VL-3B-Instruct VLM을 사용하여
이미지 기반 텍스트를 생성하고 FiLM (Feature-wise Linear Modulation)
파라미터를 생성하는 모듈을 제공합니다.

Main Components:
- VLMFiLMConditioner: 전체 파이프라인 관리
- QwenVLMTextGenerator: VLM을 사용한 텍스트 생성/임베딩
- TextToConditionVector: 텍스트 임베딩을 조건 벡터로 변환
- AdaptiveFiLMHead: 조건 벡터를 FiLM 파라미터로 변환

Usage:
    from src.conditioning import VLMFiLMConditioner, AdaptiveFiLMHead
    
    # Conditioner 초기화
    conditioner = VLMFiLMConditioner(
        enabled=True,
        cond_dim=256,
        cache_dir="cache/vlm_profiles"
    )
    
    # VLM conditioning 계산
    vlm_cond = conditioner.compute_condition(image)
    
    # FiLM 파라미터 생성
    film_head = AdaptiveFiLMHead(cond_dim=256, channels=128)
    gamma, beta = film_head(vlm_cond["cond_vec"])
    
    # FiLM 적용
    h_conditioned = (1 + gamma) * h + beta
"""

from .vlm_film_conditioner import (
    VLMConditioner,
    VLMFiLMConditioner,
    QwenVLMTextGenerator,
    TextToConditionVector,
    AdaptiveFiLMHead,
)

__all__ = [
    "VLMConditioner",
    "VLMFiLMConditioner",
    "QwenVLMTextGenerator",
    "TextToConditionVector",
    "AdaptiveFiLMHead",
]

__version__ = "0.1.0"
