# DFM+VLM 예측 문제 정리

## 1. 예측 대상: x1 mask가 맞음

- **DFM binary**: flow는 `x0`(노이즈) → `x1`(이진 세그멘테이션 마스크).
- **학습**: `logits = model(xt, t, cond)`, `loss = BCEWithLogits(logits, x1)` → 모델은 **x1에 대한 logits**를 예측하도록 학습됨.
- **추론**: sampler가 `t=1.0`에서 `logits_final = model_fn(x, t=1, cond)`, `p_final = sigmoid(logits_final)` 반환.
- **p_final** = P(x1=1) = **x1 마스크의 확률 맵**.
- **평가**: `preds = (output_geometry > 0.5).long()` 로 이진화 후 Dice 등 계산.

→ **x1 mask(세그멘테이션 마스크)를 확률로 예측하고, eval에서 0.5 기준으로 이진화하는 구조가 맞음.**

---

## 2. DFM+VLM에서 예측이 깨졌던 원인들

### (1) VLM Film decoder stages 불일치

- **상황**: 체크포인트에는 `vlm_film_heads.0~3` 4개가 있는데, hparams에는 `vlm_film_decoder_stages: None`만 저장됨.
- **코드 기본값**: `None` → `[0, 1]` (2 stage만 FiLM 적용).
- **결과**:  
  - 2 stage로 로드하면 체크포인트의 head 2개만 로드되고, 나머지 2개는 무시됨 → 학습 시 사용한 4 stage 구조와 다름.  
  - 4 stage로 로드하지 않으면 “학습 시와 다른 모델 구조”로 예측하게 됨.
- **대응**: 체크포인트 `state_dict`에서 `vlm_film_heads.*` 개수를 세어 stage 리스트를 만들고, 그걸로 `vlm_film_decoder_stages`를 넘겨 4 stage 그대로 로드하도록 수정함.

### (2) Sampler 출력 형태 (과거 이슈)

- **원래**: t=1.0에서 **binary**를 반환하거나, 중간 step의 noisy binary를 반환하는 코드가 있었을 수 있음.
- **올바른 동작**: `t=1.0`에서 **확률 맵 `p_final = sigmoid(logits_final)`**만 반환하고, 이진화는 eval에서만 `> 0.5`로 수행.
- **현재**: `sampler_dfm_euler` / `sampler_dfm_heun` 모두 `return p_final`로 확률 맵 반환하도록 되어 있음.

### (3) VLM embedding: patch 단위 vs full-image

- **학습**: `random_patch_batch`로 patch 추출 후, **patch 단위**로 `_get_vlm_film_cond(images, batch)` 호출 → patch마다 VLM embedding 계산.
- **평가**: sliding window로 patch 단위 추론. 처음에는 “full image 한 번만 embedding 계산 후 캐시”로 바꿨다가, 학습과 맞추기 위해 **patch 단위로 매번 계산**하도록 되돌림.
- **정리**: 학습과 동일하게 **sample(patch) 단위로 VLM embedding 계산**하는 것이 맞고, 그렇게 맞춰 둔 상태.

---

## 3. 현재 예측 파이프라인 (DFM+VLM)

1. **로드**: 체크포인트에서 VLM Film stage 개수 자동 감지 → 4개면 `vlm_film_decoder_stages=[0,1,2,3]`으로 로드.
2. **추론**:  
   - `_sliding_window_predict` → patch마다 `sample()` 호출.  
   - `sample()` 내부에서 `sampler_dfm_euler` 사용.  
   - sampler는 `t=1.0`에서 **x1에 대한 확률 맵 `p_final`** 반환.  
   - sliding window 결과를 merge한 뒤 `output_geometry`로 사용.
3. **이진화**: `preds = (output_geometry > 0.5).long()`.
4. **메트릭**: `preds` vs `labels`로 Dice 등 계산.

→ **예측 대상은 x1 mask가 맞고, 위와 같은 이유들(stages 불일치, sampler 출력, VLM embedding 단위)이 DFM+VLM 예측이 제대로 안 되던 원인으로 정리됨.**

---

## 4. 남은 이슈 (validation vs test Dice 차이)

- Validation Dice ≈ 0.87, Test Dice는 수정 후에도 0.12~0.26 수준으로 차이가 큼.
- stages/sampler/VLM embedding을 위와 같이 맞춘 상태에서도 차이가 남는다면, **test set 특성**, **데이터 경로/전처리**, **배치 메타데이터(image_id 등)** 차이 가능성은 추가로 확인하는 것이 좋음.
