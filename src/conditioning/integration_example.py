"""
VLM-FiLM Conditioning 통합 예시 코드.

이 파일은 기존 FlowModel에 VLM-FiLM을 통합하는 방법을 보여줍니다.
실제 프로젝트에서는 이 코드를 참고하여 최소한의 수정만 하세요.
"""

import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict

from src.conditioning.vlm_film_conditioner import VLMFiLMConditioner, AdaptiveFiLMHead


# =============================================================================
# 예시 1: FlowModel에 VLM-FiLM 추가 (Minimal Diff)
# =============================================================================

class FlowModelWithVLMFiLM(L.LightningModule):
    """
    기존 FlowModel에 VLM-FiLM conditioning을 추가한 버전.
    
    변경 사항:
    1. __init__에 use_vlm_film, vlm_film_config 인자 추가
    2. VLMFiLMConditioner와 AdaptiveFiLMHead 초기화
    3. forward에서 vlm conditioning 계산 및 FiLM 적용
    """
    
    def __init__(
        self,
        # 기존 FlowModel 인자들
        arch_name: str = "dhariwal_concat_unet",
        image_size: int = 320,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-5,
        # ... 기타 인자들 ...
        
        # 새로 추가: VLM-FiLM 관련
        use_vlm_film: bool = False,
        vlm_film_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ============ 기존 초기화 코드 ============
        # ... (기존 model, loss, metrics 등 초기화) ...
        
        # ============ VLM-FiLM 초기화 (NEW) ============
        self.use_vlm_film = use_vlm_film
        
        if use_vlm_film:
            vlm_config = vlm_film_config or {}
            
            # VLM Conditioner 초기화
            self.vlm_conditioner = VLMFiLMConditioner(
                enabled=True,
                model_name=vlm_config.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
                cond_dim=vlm_config.get("cond_dim", 256),
                cache_dir=vlm_config.get("cache_dir", "cache/vlm_profiles"),
                prompt_template=vlm_config.get("prompt_template"),
                verbose=vlm_config.get("verbose", False),
            )
            
            # Decoder의 각 stage에 대한 FiLM heads
            # NOTE: decoder_channels는 실제 decoder 구조에 맞게 조정 필요
            decoder_channels = self._get_decoder_channels()
            
            self.film_heads = nn.ModuleList([
                AdaptiveFiLMHead(
                    cond_dim=vlm_config.get("cond_dim", 256),
                    channels=ch,
                    hidden_dim=256,
                    use_difficulty_gate=True
                )
                for ch in decoder_channels
            ])
            
            print(f"[VLM-FiLM] Initialized with {len(self.film_heads)} FiLM heads")
        else:
            self.vlm_conditioner = None
            self.film_heads = None
    
    def _get_decoder_channels(self) -> list:
        """
        Decoder의 채널 수를 추출.
        실제 구현에서는 decoder 구조에 맞게 수정 필요.
        """
        # 예시: UNet-style decoder
        return [256, 128, 64, 32]
    
    def forward(self, x: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional VLM-FiLM conditioning.
        
        Args:
            x: Noisy input [B, C, H, W]
            cond_img: Conditioning image [B, C, H, W]
        
        Returns:
            output: Predicted clean mask [B, C, H, W]
        """
        
        # ============ VLM Conditioning (NEW) ============
        vlm_cond = None
        if self.use_vlm_film:
            # VLM conditioning 계산 (no grad)
            # NOTE: self.training 조건을 제거하면 inference에서도 사용
            vlm_cond = self.vlm_conditioner.compute_condition(
                image=cond_img,
                prompt=None,  # 기본 프롬프트 사용
                image_id=None  # 이미지 해시로 자동 계산
            )
        
        # ============ 기존 Encoder 로직 ============
        # ... (기존 encoder forward) ...
        # encoder_features = self.encoder(torch.cat([x, cond_img], dim=1))
        
        # ============ Decoder with optional FiLM ============
        decoder_output = x  # placeholder
        
        # 예시: Decoder loop with FiLM
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Decoder block forward (기존 코드)
            h = decoder_block(decoder_output, skip_connections[i])
            
            # ============ FiLM 적용 (NEW) ============
            if vlm_cond is not None and self.film_heads is not None:
                gamma, beta = self.film_heads[i](
                    vlm_cond["cond_vec"],
                    vlm_cond["difficulty_scalar"]
                )
                # Residual FiLM: y = (1 + gamma) * h + beta
                h = (1 + gamma) * h + beta
            
            decoder_output = h
        
        # ============ 기존 출력 로직 ============
        # ... (final conv, activation 등) ...
        
        return decoder_output
    
    def training_step(self, batch, batch_idx):
        """Training step (기존 코드와 동일)."""
        # ... (기존 training step 로직) ...
        pass
    
    def configure_optimizers(self):
        """Optimizer 설정 (기존 코드와 동일)."""
        # VLM-FiLM 파라미터도 자동으로 포함됨
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )


# =============================================================================
# 예시 2: Decoder Block 내부에 FiLM 통합
# =============================================================================

class DecoderBlockWithFiLM(nn.Module):
    """
    Decoder block with optional FiLM conditioning.
    
    기존 decoder block을 수정하는 경우 이 예시를 참고.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        vlm_film_params: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Forward with optional FiLM conditioning.
        
        Args:
            x: Input features [B, C_in, H, W]
            skip: Skip connection [B, C_skip, H, W]
            vlm_film_params: Optional (gamma, beta) tuple for FiLM
        
        Returns:
            h: Output features [B, C_out, H, W]
        """
        # Skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        
        # ============ Apply FiLM after first normalization (NEW) ============
        if vlm_film_params is not None:
            gamma, beta = vlm_film_params
            h = (1 + gamma) * h + beta
        
        h = self.activation(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h


# =============================================================================
# 예시 3: 간단한 사용 예시
# =============================================================================

def example_usage():
    """VLM-FiLM conditioner 사용 예시."""
    print("=" * 60)
    print("VLM-FiLM Conditioning 사용 예시")
    print("=" * 60)
    
    # 1. Config 설정
    vlm_film_config = {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "cond_dim": 256,
        "cache_dir": "cache/vlm_profiles",
        "verbose": True,
    }
    
    # 2. 모델 초기화 (VLM OFF)
    print("\n[1] VLM OFF 모드")
    model_off = FlowModelWithVLMFiLM(
        use_vlm_film=False,
        vlm_film_config=None,
    )
    print(f"  - use_vlm_film: {model_off.use_vlm_film}")
    print(f"  - vlm_conditioner: {model_off.vlm_conditioner}")
    
    # 3. 모델 초기화 (VLM ON)
    print("\n[2] VLM ON 모드")
    model_on = FlowModelWithVLMFiLM(
        use_vlm_film=True,
        vlm_film_config=vlm_film_config,
    )
    print(f"  - use_vlm_film: {model_on.use_vlm_film}")
    print(f"  - vlm_conditioner: {type(model_on.vlm_conditioner).__name__}")
    print(f"  - film_heads: {len(model_on.film_heads)} heads")
    
    # 4. Dummy forward (VLM OFF)
    print("\n[3] Forward pass (VLM OFF)")
    dummy_x = torch.randn(1, 1, 256, 256)
    dummy_cond = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        # output_off = model_off(dummy_x, dummy_cond)
        print(f"  - Input shape: {dummy_x.shape}")
        # print(f"  - Output shape: {output_off.shape}")
    
    # 5. Forward pass (VLM ON) - 실제로는 VLM 호출이 발생
    print("\n[4] Forward pass (VLM ON)")
    print("  - VLM이 활성화되면 이미지 분석 후 FiLM 적용")
    print("  - 첫 forward는 VLM 로딩으로 느릴 수 있음 (~500ms)")
    print("  - 이후 forward는 캐싱으로 빠름 (~5ms)")
    
    print("\n" + "=" * 60)
    print("예시 완료!")
    print("=" * 60)


# =============================================================================
# 예시 4: Config에서 모델 생성
# =============================================================================

def create_model_from_config(config: dict):
    """
    Config dict에서 모델 생성 (train.py에서 사용).
    
    Args:
        config: YAML config 파일을 로드한 dict
    
    Returns:
        model: FlowModelWithVLMFiLM 인스턴스
    """
    model_config = config['model']
    
    # VLM-FiLM 설정 추출
    use_vlm_film = model_config.get('use_vlm_film', False)
    vlm_film_config = model_config.get('vlm_film_config', {})
    
    # 모델 생성
    model = FlowModelWithVLMFiLM(
        arch_name=model_config['arch_name'],
        image_size=model_config['image_size'],
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config['weight_decay'],
        # ... 기타 인자들 ...
        
        # VLM-FiLM
        use_vlm_film=use_vlm_film,
        vlm_film_config=vlm_film_config,
    )
    
    print(f"[Model] Created with VLM-FiLM: {use_vlm_film}")
    
    return model


# =============================================================================
# 예시 5: 기존 체크포인트 로드 및 VLM-FiLM 추가
# =============================================================================

def load_and_add_vlm_film(checkpoint_path: str, vlm_film_config: dict):
    """
    기존 체크포인트를 로드하고 VLM-FiLM을 추가.
    
    주의: 새로운 파라미터(FiLM heads)가 추가되므로 fine-tuning 필요.
    
    Args:
        checkpoint_path: 기존 체크포인트 경로
        vlm_film_config: VLM-FiLM 설정
    
    Returns:
        model: VLM-FiLM이 추가된 모델
    """
    print(f"[Load] Loading checkpoint: {checkpoint_path}")
    
    # 1. 기존 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. 모델 생성 (VLM-FiLM 활성화)
    model = FlowModelWithVLMFiLM(
        use_vlm_film=True,
        vlm_film_config=vlm_film_config,
        **checkpoint['hyper_parameters']
    )
    
    # 3. 기존 가중치 로드 (strict=False: 새로운 파라미터 무시)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['state_dict'],
        strict=False
    )
    
    print(f"[Load] Missing keys (new VLM-FiLM params): {len(missing_keys)}")
    print(f"[Load] Unexpected keys: {len(unexpected_keys)}")
    
    # 4. VLM-FiLM 파라미터는 랜덤 초기화됨
    print("[Load] VLM-FiLM parameters randomly initialized")
    print("[Load] Fine-tuning recommended!")
    
    return model


if __name__ == "__main__":
    # 간단한 사용 예시 실행
    example_usage()
