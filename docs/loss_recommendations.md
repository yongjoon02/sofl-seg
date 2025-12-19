# Loss 조합 추천 가이드

## Task 특성
- **모델**: Flow Matching (continuous value prediction)
- **Target**: SAUNA soft label (boundary + thickness uncertainty)
- **데이터셋**: XCA (혈관 분할)
- **특징**: 얇은 구조, 연결성 중요, 클래스 불균형

## 추천 Loss 조합

### 🥇 추천 1: L1 + BCE (SAUNA 최적 조합) ⭐
**설명**: Flow matching 기본 + SAUNA soft label에 최적화된 BCE

```yaml
model:
  loss_type: l1_bce
  l1_weight: 1.0      # Flow matching 기본 loss (velocity field)
  bce_weight: 0.5     # SAUNA soft label에 적합한 BCE loss
```

**장점**:
- ✅ **SAUNA soft label에 가장 적합** (Supervised model에서 검증됨)
- ✅ BCE는 [0, 1] 확률값에 최적화
- ✅ Flow matching 기본 원리 유지
- ✅ 안정적인 학습

**이유**: 
- Supervised model에서 SAUNA soft label에 `bce_l2` 조합이 효과적이었음
- BCE는 확률 분포를 직접 최적화하므로 soft label과 잘 맞음
- Flow matching의 velocity field (L1) + 최종 geometry (BCE) 조합

---

### 🥈 추천 2: L1 + BCE + L2 (SAUNA + Smoothness)
**설명**: SAUNA 최적 조합 + smoothness 보장

```yaml
model:
  loss_type: l1_bce_l2
  l1_weight: 1.0
  bce_weight: 0.5
  l2_weight: 0.1      # Smoothness regularization
```

**장점**:
- ✅ SAUNA soft label 최적화 (BCE)
- ✅ Boundary smoothness 향상 (L2)
- ✅ Flow matching 기본 유지 (L1)

**단점**:
- Loss balancing 필요

---

### 🥉 추천 3: L1 + BCE + Dice (Segmentation Quality)
**설명**: SAUNA 최적 + segmentation quality 직접 최적화

```yaml
model:
  loss_type: l1_bce_dice
  l1_weight: 1.0
  bce_weight: 0.5
  dice_weight: 0.2
```

**장점**:
- ✅ SAUNA soft label 최적화 (BCE)
- ✅ Segmentation quality 직접 최적화 (Dice)
- ✅ 클래스 불균형 문제 완화 (Dice)
- ✅ Flow matching 기본 유지 (L1)

**단점**:
- Loss balancing 필요

---

### 🥉 추천 4: L1 + L2 (Smoothness만)
**설명**: Flow matching 기본 + smoothness (BCE 없이)

```yaml
model:
  loss_type: l1_l2
  l1_weight: 1.0
  l2_weight: 0.1
```

**장점**:
- ✅ Flow matching 기본 원리 유지
- ✅ L2로 boundary smoothness 향상
- ✅ 안정적인 학습

**단점**:
- SAUNA soft label에 최적화되지 않음
- Segmentation quality를 직접 최적화하지 않음

---

### 🥉 추천 5: L1 + Topo (연결성 강조)
**설명**: Flow matching + topology preservation

```yaml
model:
  loss_type: l1_topo
  l1_weight: 1.0
  topo_weight: 0.1
  topo_maxdim: 1      # Betti0 + Betti1 (components + loops)
```

**장점**:
- ✅ 혈관 연결성 보장
- ✅ Topology-aware 학습
- ✅ 얇은 구조물에 유리

**단점**:
- TopoLoss는 binary mask 필요 (soft label과 직접 호환 어려움)
- 계산 비용 높음
- Validation에서는 제외 권장

**주의**: TopoLoss는 soft label과 직접 사용하기 어려우므로, 
validation loss로만 사용하거나 threshold 후 사용

---

### 🎯 추천 6: L1 + BCE + L2 + Dice (종합)
**설명**: 모든 요소를 포함한 종합 조합

```yaml
model:
  loss_type: l1_l2_dice_topo
  l1_weight: 1.0
  l2_weight: 0.1
  dice_weight: 0.2
  topo_weight: 0.05   # 작은 weight로 시작
  topo_maxdim: 1
```

**장점**:
- ✅ 모든 측면 고려
- ✅ 최고 성능 가능성

**단점**:
- Loss balancing 복잡
- 학습 불안정 가능성
- 계산 비용 높음

---

## Loss 구현 방법

### FlowModel에 loss 조합 추가 필요

현재 FlowModel은 단순히 L1 loss만 사용:
```python
loss = torch.abs(v - ut).mean()
```

다음과 같이 확장 가능:

```python
# L1 (기본)
l1_loss = torch.abs(v - ut).mean()

# L2 (smoothness)
l2_loss = ((v - ut) ** 2).mean()

# Dice (segmentation quality)
# output_geometry를 threshold하여 사용
pred_binary = (output_geometry > 0.5).float()
dice_loss = dice_loss_fn(pred_binary, geometry)

# Total loss
loss = l1_loss + 0.1 * l2_loss + 0.2 * dice_loss
```

---

## 실험 순서 추천

1. **1단계**: **L1 + BCE** (추천 1) ⭐ **가장 추천**
   - SAUNA soft label에 최적화
   - Supervised model에서 검증된 조합
   - 빠른 baseline 확립

2. **2단계**: L1 + BCE + L2 (추천 2)
   - Smoothness 추가
   - Boundary quality 향상

3. **3단계**: L1 + BCE + Dice (추천 3)
   - Segmentation quality 직접 최적화
   - 클래스 불균형 완화

4. **4단계**: L1 + Topo (추천 5)
   - 연결성 개선 확인
   - TopoLoss weight 작게 시작 (0.01-0.1)

5. **5단계**: 종합 조합 (추천 6)
   - 최고 성능 도전
   - 신중한 weight 튜닝 필요

---

## Loss Weight 튜닝 가이드

### 초기 값
```yaml
l1_weight: 1.0        # 기준 (항상 1.0)
bce_weight: 0.5       # SAUNA soft label에 적합 (Supervised model 참고)
l2_weight: 0.1        # 작게 시작
dice_weight: 0.2      # 중간
topo_weight: 0.05     # 매우 작게 시작
```

### 튜닝 원칙
1. **L1은 항상 1.0**: Flow matching의 기본
2. **BCE는 0.3-0.7**: SAUNA soft label에 중요 (Supervised model: 0.5-1.0)
3. **L2는 0.05-0.2**: 너무 크면 over-smooth
4. **Dice는 0.1-0.5**: Segmentation quality에 따라 조정
5. **Topo는 0.01-0.1**: 매우 작게 시작, 점진적 증가

### 모니터링 지표
- `train/l1_loss`: 기본 flow matching loss (velocity field)
- `train/bce_loss`: SAUNA soft label 최적화 loss ⭐
- `train/l2_loss`: Smoothness loss
- `train/dice_loss`: Segmentation quality
- `train/topo_loss`: Topology loss
- `val/dice`: 최종 성능 지표

---

## 주의사항

1. **SAUNA soft label**: Continuous values [0, 1]이므로 L1/L2와 잘 맞음
2. **TopoLoss**: Binary mask 필요하므로 threshold 후 사용
3. **Loss scale**: 각 loss의 scale이 다를 수 있으므로 normalization 고려
4. **Validation**: TopoLoss는 validation에서 제외하여 속도 향상

---

## 최종 추천

**⭐ 초기 실험 (가장 추천)**: **L1 + BCE** (추천 1)
- **SAUNA soft label에 최적화** (Supervised model에서 검증됨)
- Flow matching 기본 유지
- 빠른 baseline 확립
- **BCE는 [0, 1] 확률값에 최적화되어 SAUNA와 완벽히 맞음**

**성능 향상 필요시**: **L1 + BCE + L2** (추천 2)
- SAUNA 최적화 + smoothness
- Boundary quality 향상

**Segmentation quality 중요시**: **L1 + BCE + Dice** (추천 3)
- SAUNA 최적화 + segmentation quality 직접 최적화
- 클래스 불균형 완화

**연결성 중요시**: **L1 + Topo** (추천 5)
- 혈관 연결성 보장
- 얇은 구조물에 유리
- 주의: TopoLoss는 binary mask 필요

