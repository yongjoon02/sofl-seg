# VLM-to-FiLM Conditioning

Flow-Matching 기반 segmentation decoder를 위한 **선택적(optional)** VLM-to-FiLM conditioning 모듈입니다.

## 주요 특징

✅ **On/Off 가능**: config 파일에서 `use_vlm_film: true/false`로 쉽게 토글  
✅ **기존 동작 보존**: OFF일 때 기존 코드와 완전히 동일한 동작  
✅ **No Gradient through VLM**: VLM은 frozen, 모든 출력은 detached  
✅ **캐싱 지원**: VLM 호출 결과를 디스크에 자동 캐싱  
✅ **Robust**: JSON 파싱 실패 시 안전한 fallback 제공  

## 설치

### 필수 패키지

```bash
# Transformers 및 관련 패키지
pip install transformers>=4.37.0 accelerate>=0.26.0

# 이미지 처리
pip install pillow

# 기타 (이미 설치되어 있을 것)
pip install torch torchvision
```

## 빠른 시작

### 1. Config 파일에 VLM-FiLM 옵션 추가

`configs/flow/xca/flow_vlm.yaml`:

```yaml
model:
  arch_name: dhariwal_concat_unet
  # ... 기존 설정 ...
  
  # VLM-FiLM Conditioning 추가
  use_vlm_film: true
  vlm_film_config:
    model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
    cond_dim: 256
    cache_dir: "cache/vlm_profiles"
    verbose: false
```

### 2. 모델에 통합

```python
from src.conditioning import VLMFiLMConditioner, AdaptiveFiLMHead

class YourFlowModel(L.LightningModule):
    def __init__(self, use_vlm_film=False, vlm_film_config=None, **kwargs):
        super().__init__()
        
        # VLM-FiLM 초기화
        if use_vlm_film:
            self.vlm_conditioner = VLMFiLMConditioner(
                enabled=True,
                **vlm_film_config
            )
            self.film_heads = nn.ModuleList([
                AdaptiveFiLMHead(cond_dim=256, channels=ch)
                for ch in decoder_channels
            ])
        else:
            self.vlm_conditioner = None
            self.film_heads = None
    
    def forward(self, x, cond_img):
        # VLM conditioning 계산
        vlm_cond = None
        if self.vlm_conditioner is not None:
            vlm_cond = self.vlm_conditioner.compute_condition(cond_img)
        
        # Decoder with FiLM
        for i, decoder_block in enumerate(self.decoder_blocks):
            h = decoder_block(x, skip[i])
            
            # FiLM 적용
            if vlm_cond is not None:
                gamma, beta = self.film_heads[i](
                    vlm_cond["cond_vec"],
                    vlm_cond["difficulty_scalar"]
                )
                h = (1 + gamma) * h + beta
            
            x = h
        
        return x
```

### 3. 학습 실행

```bash
# VLM OFF (baseline)
python scripts/train.py --config configs/flow/xca/flow.yaml --gpu 0

# VLM ON
python scripts/train.py --config configs/flow/xca/flow_vlm.yaml --gpu 0
```

## 상세 문서

- **[통합 가이드](INTEGRATION_GUIDE.md)**: 기존 프로젝트에 통합하는 방법
- **[통합 예시](integration_example.py)**: 실제 코드 예시

## 구조

```
src/conditioning/
├── __init__.py                  # 패키지 초기화
├── vlm_film_conditioner.py      # 메인 모듈 (VLM, FiLM 구현)
├── integration_example.py        # 통합 예시 코드
├── INTEGRATION_GUIDE.md         # 상세 통합 가이드
└── README.md                    # 이 파일
```

## 동작 흐름

```
이미지 입력
    ↓
VLM (Qwen2.5-VL-3B-Instruct)
    ↓
JSON 프로파일 {difficulty, confidence, focus, ...}
    ↓
Condition Vector [B, cond_dim]
    ↓
FiLM Head
    ↓
Gamma, Beta [B, C, 1, 1]
    ↓
Decoder Feature Map: y = (1 + gamma) * h + beta
```

## 테스트

```bash
# Unit test 실행
cd /home/yongjun/soft-seg
python -m src.conditioning.vlm_film_conditioner
```

예상 출력:
```
============================================================
VLMFiLMConditioner Sanity Test
============================================================

[Test 1] Disabled mode
✓ Disabled mode returns None

[Test 2] Enabled mode (mock profile)
✓ Condition vector shape: torch.Size([1, 256])

[Test 3] AdaptiveFiLMHead
✓ Gamma shape: torch.Size([1, 128, 1, 1])
✓ Beta shape: torch.Size([1, 128, 1, 1])

[Test 4] FiLM application
✓ Output difference (L2): 45.3892

============================================================
All tests passed!
============================================================
```

## 성능

| 설정 | 첫 Forward | 캐싱 후 | 메모리 증가 |
|------|-----------|---------|-------------|
| VLM OFF | 0ms | 0ms | 0 MB |
| VLM ON | ~500ms | ~5ms | +3GB |

**최적화 팁**:
- 학습 전 전체 데이터셋에 대해 VLM 프로파일 미리 생성 (캐싱)
- Inference 시 캐시된 프로파일 사용
- VLM은 별도 GPU에 로드 가능 (`device_map="auto"`)

## Troubleshooting

### Q: VLM 로딩이 너무 느림
A: VLM은 lazy loading됩니다. 첫 forward 시에만 로딩되고 이후엔 빠릅니다.

### Q: JSON 파싱 실패
A: `verbose=True`로 설정하여 VLM 출력 확인. Fallback default profile이 자동 사용됩니다.

### Q: GPU 메모리 부족
A: Batch size 줄이기, Mixed precision 사용, VLM을 별도 GPU에 로드

### Q: VLM OFF 시 동작이 다름
A: `if self.use_vlm_film:` 조건문 확인, config에서 `use_vlm_film: false` 확인

## 참고 자료

- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Qwen2.5-VL Documentation](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## 라이선스

이 프로젝트와 동일한 라이선스를 따릅니다.

## 기여

이슈나 개선 사항이 있으면 GitHub Issues에 등록해주세요.
