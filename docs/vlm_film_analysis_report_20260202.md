# VLM-FiLM 통합 현황 분석 리포트

**작성일**: 2026년 2월 2일  
**프로젝트**: SOFL-Seg (Vessel Segmentation with Flow Matching)  
**분석 대상**: VLM-FiLM 조건부 모듈 통합 효과

---

## 📋 요약 (Executive Summary)

VLM-FiLM(Vision-Language Model Feature-wise Linear Modulation)을 Flow Matching 기반 혈관 세그멘테이션 모델에 통합하여 의미론적 조건부 생성 성능을 평가했습니다. 

**핵심 결론**: 현재 VLM-FiLM 설정으로는 **기본 Flow Matching 모델 대비 통계적으로 유의미한 성능 향상을 확인하지 못했습니다** (Dice Score 차이 < 0.3%). 추가 하이퍼파라미터 최적화가 필요합니다.

---

## 🔬 실험 구성

### 실험 환경
- **데이터셋**: XCA (X-ray Coronary Angiography)
  - Train: 155 samples
  - Validation: 20 samples  
  - Test: 46 samples
- **모델 아키텍처**: MedSegDiff UNet + Discrete Flow Matching (DFM)
- **VLM 모델**: Qwen2.5-VL-3B-Instruct
- **학습 설정**:
  - Learning rate: 0.0002
  - Batch size: 6
  - Max epochs: 500
  - Timesteps: 50

### 평가 대상 실험

| Experiment ID | VLM-FiLM 적용 | 설명 |
|--------------|---------------|------|
| **20260128_105314** | ❌ **없음** | 기본 Flow Matching 모델 (baseline) |
| **20260131_174749** | ✅ **Stage 0, 1** | 디코더 저해상도 레이어만 적용 (128ch, 96ch) |
| **20260129_174302** | ✅ **전체 (추정)** | 디코더 전체 레이어 적용 |

---

## 📊 실험 결과

### Test Set 성능 비교 (46 samples)

| 메트릭 | Baseline<br>(VLM 없음) | VLM Stage 0,1 | VLM 전체? | 최고 성능 |
|--------|------------------------|---------------|-----------|-----------|
| **Dice Score** | 0.8631 | 0.8623 | **0.8653** | VLM 전체 (+0.22%) |
| **IoU** | 0.7610 | 0.7598 | **0.7643** | VLM 전체 (+0.33%) |
| **Precision** | **0.8985** | 0.8772 | 0.8822 | Baseline |
| **Recall** | 0.8334 | 0.8509 | **0.8517** | VLM 전체 (+1.83%) |
| **Specificity** | **0.9909** | 0.9884 | 0.9888 | Baseline |
| **clDice** | 0.8604 | 0.8579 | **0.8643** | VLM 전체 (+0.39%) |
| **Betti-0 Error** | 4.48 | 4.22 | **3.78** | VLM 전체 |
| **Betti-1 Error** | **50.09** | 52.07 | 62.39 | Baseline |

### Validation Set 성능

| 메트릭 | Baseline | VLM Stage 0,1 | VLM 전체? |
|--------|----------|---------------|-----------|
| **Dice Score** | 미기록 | 0.8747 | 0.8744 |
| **IoU** | 미기록 | 0.7783 | 0.7778 |
| **clDice** | 미기록 | 0.8507 | 0.8503 |

---

## 🔍 주요 발견사항

### 1. VLM-FiLM의 미미한 효과

**통계적 유의성 결여**:
- 세 모델 간 Dice Score 차이: **0.30%** (0.8623 ~ 0.8653)
- 이는 랜덤 변동 범위 내에 있을 가능성 높음
- 계산 비용 대비 성능 이득이 불명확

### 2. Trade-off 패턴

VLM-FiLM 추가 시:
- ✅ **Recall 향상**: 0.8334 → 0.8517 (+1.83%)
  - 더 많은 혈관 픽셀 탐지
- ❌ **Precision 하락**: 0.8985 → 0.8822 (-1.63%)
  - False positive 증가
- ❌ **위상학적 정확도 하락**: Betti-1 error 50.09 → 62.39 (+24.5%)
  - 혈관 루프 구조 보존 능력 저하

### 3. 디코더 적용 위치 영향

- **Stage 0, 1 (저해상도)**: 거의 baseline과 동일
- **전체 스테이지**: 0.22% Dice 향상 (미미함)
- **결론**: 적용 위치가 성능에 결정적 영향을 주지 않음

---

## 🛠️ 기술적 구현 현황

### 구현 완료 사항

1. **VLM-FiLM 모듈 통합** ✅
   - `src/archs/flow_model_vlm_film.py`: VLM-FiLM 지원 Flow Model
   - `src/archs/components/vlm_film.py`: AdaptiveFiLMHead 구현
   - `src/conditioning/vlm_film_conditioner.py`: Qwen2.5-VL 기반 조건부 생성기

2. **Configurable Decoder Stage 선택** ✅
   - `vlm_film_decoder_stages` 파라미터 추가
   - 임의의 디코더 레이어 조합 선택 가능 (예: [0,1], [2,3], [0,1,2,3])
   - 각 스테이지별 독립적인 FiLM head 생성

3. **Training Runner 업데이트** ✅
   - `src/runner/train_runner_vlm_film.py`: VLM-FiLM 전용 runner
   - YAML config에서 모든 VLM 파라미터 지원

4. **체크포인트 호환성** ✅
   - `load_state_dict` override로 FiLM head 개수 불일치 처리
   - 기존 체크포인트 → 새 설정 로딩 가능

### 현재 설정

```yaml
use_vlm_film: true
vlm_film_decoder_stages: [2, 3]  # 고해상도 스테이지 (64ch, 32ch)
vlm_film_config:
  model_name: "Qwen/Qwen2.5-VL-3B-Instruct"
  cond_dim: 256
  gamma_scale: 0.1
  beta_scale: 0.1
  cond_layernorm: true
```

---

## ⚠️ 문제점 및 한계

### 1. 성능 향상 미달

**예상**: VLM의 의미론적 정보가 세그멘테이션 품질 향상  
**현실**: 0.22% Dice 향상 (통계적으로 무의미)

**가능한 원인**:
- VLM 특징이 Flow Matching의 continuous trajectory와 충돌
- FiLM modulation이 flow field를 왜곡
- Gamma/Beta scale이 부적절 (너무 강하거나 약함)

### 2. 위상학적 정확도 저하

- Betti-1 error가 50 → 62로 **24.5% 증가**
- 혈관 네트워크의 루프 구조 보존 능력 악화
- 의학적 응용에서 치명적 문제 가능성

### 3. Precision-Recall Trade-off

- Recall 향상 대신 Precision 희생
- Over-segmentation 경향 (false positive 증가)
- 임상 적용 시 추가 후처리 필요

### 4. 계산 비용 증가

- VLM inference: Qwen2.5-VL-3B (~3B 파라미터)
- 메모리 사용량 증가
- 학습/추론 시간 증가
- **비용 대비 효과 불분명**

---

## 💡 향후 개선 방향

### 1. 하이퍼파라미터 최적화

**FiLM Scale 조정**:
```yaml
# 현재 설정
gamma_scale: 0.1
beta_scale: 0.1

# 제안 (더 약한 modulation)
gamma_scale: 0.01  # 1/10로 감소
beta_scale: 0.01
```

**이유**: 현재 modulation이 너무 강해서 flow trajectory를 과도하게 왜곡할 가능성

### 2. 적용 위치 변경

**현재**: block1 이후, concat2 이전  
**대안**:
- Attention 이후 적용
- 디코더 최종 출력에만 적용
- Skip connection에 적용

### 3. VLM 업데이트 전략

**현재**:
```python
vlm_update_interval: 50  # 50 step마다 업데이트
```

**제안**:
- 1 step마다 업데이트 (항상 최신 상태)
- Epoch 단위 업데이트 (안정성 향상)
- Adaptive update (loss 기반)

### 4. Loss 함수 개선

**현재**: L2 + BCE + Dice  
**제안**:
- VLM alignment loss 추가
- Perceptual loss (VLM feature space)
- Topological loss (Betti number preservation)

### 5. 대체 VLM 모델 시도

| 모델 | 파라미터 | 장점 |
|------|---------|------|
| **CLIP ViT-B/16** | 150M | 경량, 빠름 |
| **BiomedCLIP** | 150M | 의료 도메인 특화 |
| **RadImageNet** | Varies | X-ray 사전학습 |

---

## 📈 비교: 다른 방법론들

### Baseline 대비 성능

| 방법 | Dice | clDice | Betti-1 Error | 특징 |
|------|------|--------|---------------|------|
| **Flow Matching (baseline)** | **0.8631** | 0.8604 | **50.09** | 빠름, 안정적 |
| **+ VLM-FiLM (Stage 0,1)** | 0.8623 | 0.8579 | 52.07 | 복잡함 |
| **+ VLM-FiLM (전체)** | 0.8653 | **0.8643** | 62.39 | 가장 복잡 |
| BerDiff (참고) | ~0.84 | ~0.82 | - | Diffusion 기반 |
| CSNet (참고) | ~0.82 | ~0.80 | - | Supervised |

**결론**: VLM-FiLM은 baseline 대비 명확한 우위 없음

---

## 🎯 권장 조치

### 즉각 조치 (Immediate)

1. **현재는 VLM 없는 baseline 사용 권장**
   - Dice: 0.8631 (충분히 높음)
   - 계산 효율적
   - 위상학적 정확도 우수

2. **VLM-FiLM 파라미터 튜닝 실험 진행**
   - Scale: [0.001, 0.01, 0.05, 0.1]
   - 적용 위치: [post-block1, post-attention, output-only]
   - 업데이트 주기: [1, 10, 50, epoch]

### 중기 계획 (1-2주)

3. **경량 VLM 모델 시도**
   - BiomedCLIP 또는 CLIP으로 교체
   - 계산 비용 감소 + 성능 검증

4. **대체 조건부 생성 방법 탐색**
   - Cross-attention 기반 conditioning
   - Adaptive Instance Normalization (AdaIN)
   - ControlNet-style guidance

### 장기 방향 (1개월+)

5. **도메인 특화 VLM 사전학습**
   - XCA 데이터로 VLM fine-tuning
   - Medical image-text pair로 사전학습

6. **Multi-modal fusion 전략**
   - VLM + Flow Matching의 앙상블
   - Late fusion vs Early fusion 비교

---

## 📚 관련 파일 및 설정

### 주요 소스 코드

```
src/archs/
├── flow_model_vlm_film.py          # VLM-FiLM 지원 Flow Model
├── components/
│   ├── vlm_film.py                 # FiLM head 구현
│   ├── diffusion_unet_vlm_film.py  # VLM-FiLM UNet wrapper
│   └── medsegdiff_flow_vlm_film.py # MedSegDiff backbone wrapper
src/conditioning/
└── vlm_film_conditioner.py         # Qwen2.5-VL conditioner
src/runner/
└── train_runner_vlm_film.py        # Training runner
```

### 설정 파일

```
configs/flow/xca/
└── flow_sauna_medsegdiff.yaml      # 현재 VLM-FiLM 설정
```

### 실험 결과

```
experiments/medsegdiff_flow/xca/
├── medsegdiff_flow_xca_20260128_105314/  # Baseline (VLM 없음)
├── medsegdiff_flow_xca_20260131_174749/  # VLM Stage 0,1
└── medsegdiff_flow_xca_20260129_174302/  # VLM 전체

results/evaluation/
├── evaluation_*_20260128_105314_xca.csv
├── evaluation_*_20260131_174749_xca.csv
└── evaluation_*_20260129_174302_xca.csv
```

---

## 🔬 실험 재현 방법

### Baseline (VLM 없음)
```bash
# configs/flow/xca/baseline.yaml 사용
bash scripts/train.sh \
  --config configs/flow/xca/flow_baseline.yaml \
  --gpu 0
```

### VLM-FiLM (Stage 0,1)
```bash
# vlm_film_decoder_stages: [0, 1] 설정
bash scripts/train.sh \
  --config configs/flow/xca/flow_sauna_medsegdiff.yaml \
  --gpu 0
```

### 평가
```bash
uv run python scripts/evaluate.py \
  --data xca \
  --models medsegdiff_flow \
  --checkpoint experiments/.../checkpoints/best.ckpt \
  --gpu 0 \
  --save-predictions
```

---

## 📝 결론

현재 VLM-FiLM 통합은 **기술적으로 성공**했으나, **성능 개선 측면에서는 실패**했습니다:

### ✅ 성공한 부분
- VLM-FiLM 모듈 완전 통합
- Configurable decoder stage 선택
- 체크포인트 호환성 확보
- 안정적인 학습 프로세스

### ❌ 개선 필요 부분
- 통계적으로 유의미한 성능 향상 없음 (Dice +0.22%)
- 위상학적 정확도 오히려 저하 (Betti-1 error +24.5%)
- 계산 비용 대비 효과 미미
- Precision-Recall trade-off 불리

### 🎯 핵심 권장사항

**당장은 VLM 없는 baseline을 사용하고**, 동시에 다음 최적화를 진행:
1. FiLM scale 조정 (0.01 또는 0.001)
2. 경량 VLM 모델 시도 (BiomedCLIP)
3. 적용 위치 변경 (attention 이후)

이후 재평가를 통해 VLM-FiLM의 실질적 가치를 재검증해야 합니다.

---

**보고서 작성**: GitHub Copilot  
**실험 수행**: 2026년 1월 28일 ~ 2월 2일  
**데이터셋**: XCA (X-ray Coronary Angiography), 46 test samples
