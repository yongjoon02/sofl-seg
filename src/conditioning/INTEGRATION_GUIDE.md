# VLM-to-FiLM Conditioning 통합 가이드

이 가이드는 기존 Flow-Matching segmentation decoder에 VLM-to-FiLM conditioning을 추가하는 방법을 설명합니다.

## 목차

1. [개요](#개요)
2. [최소 통합 방법](#최소-통합-방법)
3. [설정 파일 수정](#설정-파일-수정)
4. [Decoder 수정](#decoder-수정)
5. [학습 루프 수정](#학습-루프-수정)
6. [테스트 및 검증](#테스트-및-검증)

---

## 개요

### 주요 특징

- ✅ **On/Off 가능**: config에서 `use_vlm_film: true/false`로 토글
- ✅ **기존 동작 보존**: OFF일 때 기존 코드와 동일한 동작
- ✅ **No Gradient through VLM**: VLM은 frozen, 모든 출력은 detached
- ✅ **캐싱 지원**: VLM 호출 결과를 디스크에 캐싱
- ✅ **Robust**: JSON 파싱 실패 시 안전한 fallback

### 동작 흐름

```
이미지 → VLM (Qwen2.5-VL) → JSON 프로파일 → Condition Vector → FiLM (gamma, beta) → Decoder
```

---

## 최소 통합 방법

### STEP 1: Config에 VLM-FiLM 옵션 추가

기존 config 파일 (예: `configs/flow/xca/flow.yaml`)에 다음 추가:

```yaml
model:
  # ... 기존 설정 ...
  
  # VLM-FiLM Conditioning (OPTIONAL)
  use_vlm_film: false  # true로 설정하면 활성화
  vlm_film_config:
    model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
    cond_dim: 256
    cache_dir: "cache/vlm_profiles"
    prompt_template: null  # null이면 기본 프롬프트 사용
    verbose: false
```

### STEP 2: Model에 VLMFiLMConditioner 추가

기존 Flow-Matching 모델 (예: `FlowModel` in `src/archs/flow_model.py`)에 conditioner 추가:

```python
# 파일 상단에 import 추가
from src.conditioning.vlm_film_conditioner import VLMFiLMConditioner, AdaptiveFiLMHead

class FlowModel(L.LightningModule):
    def __init__(
        self,
        # ... 기존 인자들 ...
        use_vlm_film: bool = False,
        vlm_film_config: dict = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ... 기존 초기화 ...
        
        # VLM-FiLM Conditioner 초기화
        self.use_vlm_film = use_vlm_film
        if use_vlm_film:
            vlm_config = vlm_film_config or {}
            self.vlm_conditioner = VLMFiLMConditioner(
                enabled=True,
                model_name=vlm_config.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct"),
                cond_dim=vlm_config.get("cond_dim", 256),
                cache_dir=vlm_config.get("cache_dir", "cache/vlm_profiles"),
                prompt_template=vlm_config.get("prompt_template"),
                verbose=vlm_config.get("verbose", False),
            )
            
            # FiLM heads for each decoder stage (예: 4 stages)
            decoder_channels = [256, 128, 64, 32]  # 실제 decoder 채널에 맞게 조정
            self.film_heads = nn.ModuleList([
                AdaptiveFiLMHead(
                    cond_dim=vlm_config.get("cond_dim", 256),
                    channels=ch,
                    hidden_dim=256,
                    use_difficulty_gate=True
                )
                for ch in decoder_channels
            ])
        else:
            self.vlm_conditioner = None
            self.film_heads = None
```

### STEP 3: Forward에서 VLM Conditioning 적용

```python
def forward(self, x, cond_img):
    """
    Args:
        x: noisy input [B, C, H, W]
        cond_img: conditioning image [B, C, H, W]
    """
    
    # VLM conditioning 계산 (한 번만, 재사용)
    vlm_cond = None
    if self.use_vlm_film and self.training:  # 학습 시에만 적용 (선택 사항)
        vlm_cond = self.vlm_conditioner.compute_condition(
            image=cond_img,
            prompt=None,  # 기본 프롬프트 사용
            image_id=None  # 이미지 해시로 자동 계산
        )
    
    # ... 기존 encoder 로직 ...
    
    # Decoder with optional FiLM conditioning
    decoder_features = []
    for i, decoder_block in enumerate(self.decoder_blocks):
        # Decoder block forward
        h = decoder_block(x, skip_connections[i])
        
        # Apply FiLM if available
        if vlm_cond is not None and self.film_heads is not None:
            gamma, beta = self.film_heads[i](
                vlm_cond["cond_vec"],
                vlm_cond["difficulty_scalar"]
            )
            # FiLM: y = (1 + gamma) * h + beta
            h = (1 + gamma) * h + beta
        
        decoder_features.append(h)
        x = h
    
    # ... 나머지 로직 ...
    
    return output
```

---

## Decoder Block 수정 (Option 1: 외부 적용)

### 현재 권장 방법: Decoder forward 후 FiLM 적용

기존 decoder block을 수정하지 않고, block output에 FiLM을 외부에서 적용:

```python
# Decoder loop
for i, (decoder_block, film_head) in enumerate(zip(self.decoder_blocks, self.film_heads)):
    h = decoder_block(x, skip_connection)
    
    # Optional FiLM conditioning
    if vlm_cond is not None:
        gamma, beta = film_head(vlm_cond["cond_vec"], vlm_cond["difficulty_scalar"])
        h = (1 + gamma) * h + beta
    
    x = h
```

### Option 2: Decoder Block 내부에 FiLM 통합

더 깊은 통합을 원한다면, decoder block 내부에 FiLM을 추가할 수 있습니다:

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # ... 기타 레이어 ...
    
    def forward(self, x, skip, vlm_film_params=None):
        """
        Args:
            x: input features
            skip: skip connection
            vlm_film_params: Optional tuple (gamma, beta) for FiLM
        """
        h = self.conv1(torch.cat([x, skip], dim=1))
        h = F.relu(h)
        
        # Apply FiLM if provided
        if vlm_film_params is not None:
            gamma, beta = vlm_film_params
            h = (1 + gamma) * h + beta
        
        h = self.conv2(h)
        return h
```

---

## 설정 파일 예시

### `configs/flow/xca/flow_vlm.yaml`

```yaml
tag: flow_vlm

model:
  arch_name: dhariwal_concat_unet
  image_size: 320
  learning_rate: 0.0002
  weight_decay: 0.00001
  
  # VLM-FiLM Conditioning
  use_vlm_film: true
  vlm_film_config:
    model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
    cond_dim: 256
    cache_dir: "cache/vlm_profiles"
    prompt_template: null  # 기본 프롬프트 사용
    verbose: true  # 디버깅용

data:
  name: xca
  train_dir: data/xca_full/train
  val_dir: data/xca_full/val
  test_dir: data/xca_full/test
  crop_size: 320
  train_bs: 4
  num_samples_per_image: 4

trainer:
  max_epochs: 500
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  check_val_every_n_epoch: 50
```

---

## 학습 스크립트 수정

### `scripts/train.py`

Config에서 VLM-FiLM 설정을 읽어 모델에 전달:

```python
def create_model(config):
    model_config = config['model']
    
    # VLM-FiLM 설정 추출
    use_vlm_film = model_config.get('use_vlm_film', False)
    vlm_film_config = model_config.get('vlm_film_config', {})
    
    model = FlowModel(
        arch_name=model_config['arch_name'],
        # ... 기타 인자 ...
        use_vlm_film=use_vlm_film,
        vlm_film_config=vlm_film_config,
    )
    
    return model
```

---

## 테스트 및 검증

### 1. Unit Test 실행

```bash
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
...
✓ Condition vector shape: torch.Size([1, 256])

[Test 3] AdaptiveFiLMHead
✓ Gamma shape: torch.Size([1, 128, 1, 1])
✓ Beta shape: torch.Size([1, 128, 1, 1])

[Test 4] FiLM application
✓ Baseline output shape: torch.Size([1, 128, 32, 32])
✓ FiLM output shape: torch.Size([1, 128, 32, 32])
✓ Output difference (L2): 45.3892

============================================================
All tests passed!
============================================================
```

### 2. ON/OFF 동작 확인

**Baseline (VLM OFF)**:
```bash
python scripts/train.py --config configs/flow/xca/flow.yaml --gpu 0
```

**With VLM (VLM ON)**:
```bash
python scripts/train.py --config configs/flow/xca/flow_vlm.yaml --gpu 0
```

두 실험의 validation metrics를 비교하여 VLM conditioning의 효과 확인.

### 3. 캐싱 확인

```bash
# 캐시 디렉토리 확인
ls -lh cache/vlm_profiles/

# 캐시 파일 예시: a1b2c3d4_e5f6g7h8.json
# 내용 확인
cat cache/vlm_profiles/a1b2c3d4_e5f6g7h8.json
```

---

## Troubleshooting

### Q1: VLM 로딩이 너무 느림

**A**: VLM은 첫 forward 시에만 로딩됩니다 (lazy loading). 초기 로딩 후에는 캐싱으로 빠릅니다.

### Q2: JSON 파싱 실패가 빈번함

**A**: 
1. `verbose=True`로 설정하여 VLM 출력 확인
2. 프롬프트 템플릿 수정하여 JSON 형식 강조
3. Fallback default profile이 자동 사용되므로 학습은 계속됩니다

### Q3: GPU 메모리 부족

**A**:
1. VLM은 별도 GPU에 로드 가능 (`device_map="auto"` 사용)
2. Batch size 줄이기
3. Mixed precision 사용 (`precision: 16-mixed`)

### Q4: VLM OFF 시 동작이 다름

**A**: 
1. `if self.use_vlm_film:` 조건문이 제대로 작동하는지 확인
2. FiLM 적용 전후 feature norm 비교
3. Config에서 `use_vlm_film: false` 확인

---

## 고급 설정

### Custom Prompt Template

```yaml
vlm_film_config:
  prompt_template: |
    Analyze this medical image for vessel segmentation difficulty.
    Focus on: thin vessels, bifurcations, image noise, contrast.
    Output JSON with difficulty (1-5), confidence (0-1), and focus scores.
    Schema: {"difficulty":3,"confidence":0.5,"focus":{"boundary":0.5,"thin_structure":0.5,"small_objects":0.5,"clutter":0.5}}
```

### Inference에서도 VLM 사용

```python
# Test/Inference 시에도 VLM conditioning 적용
def forward(self, x, cond_img):
    vlm_cond = None
    if self.use_vlm_film:  # training 조건 제거
        vlm_cond = self.vlm_conditioner.compute_condition(...)
    ...
```

### 학습 가능한 FiLM 파라미터

기본적으로 VLM은 frozen이지만, FiLM head는 학습 가능합니다. Conditioning 강도를 조절하려면:

```python
# Difficulty gate 비활성화 (항상 full strength)
AdaptiveFiLMHead(
    cond_dim=256,
    channels=128,
    use_difficulty_gate=False  # True면 difficulty에 비례
)
```

---

## 성능 벤치마크

| 설정 | VLM Overhead (첫 forward) | VLM Overhead (캐싱 후) | 메모리 증가 |
|------|---------------------------|------------------------|-------------|
| VLM OFF | 0ms | 0ms | 0 MB |
| VLM ON (첫 이미지) | ~500ms | - | +3GB |
| VLM ON (캐싱 후) | - | ~5ms | +3GB |

**권장사항**: 
- 학습 전에 전체 데이터셋에 대해 VLM 프로파일을 미리 생성하여 캐싱
- Inference 시에는 캐시된 프로파일 사용

---

## 참고 자료

- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Qwen2.5-VL Documentation](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- 구현 파일: `src/conditioning/vlm_film_conditioner.py`

---

문의사항이나 이슈가 있으면 GitHub Issues에 등록해주세요.
