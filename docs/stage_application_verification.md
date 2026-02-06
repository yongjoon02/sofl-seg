# VLM FiLM Decoder Stages 적용 검증

## 📋 검사 항목

Config의 `vlm_film_decoder_stages` 설정이 Training → Validation → Evaluation 전체에 일관되게 적용되는지 검증

---

## ✅ 1. Config 파일 → Training Runner

**파일**: `src/runner/train_runner_vlm_film.py` (line 59)

```python
vlm_film_decoder_stages=self.model_cfg.get('vlm_film_decoder_stages', None),
```

**상태**: ✅ **정상**
- Config의 `vlm_film_decoder_stages` 값을 그대로 모델 생성자에 전달

---

## ✅ 2. Training: 모델 초기화 시 Stage 설정

**파일**: `src/archs/flow_model_vlm_film.py` (lines 219-230)

```python
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
```

**상태**: ✅ **정상**
- Config 값을 `self._vlm_film_stage_indices`에 저장
- 해당 stage에 맞는 FiLM head만 생성
- Config 없으면 기본값 [0, 1] 사용

---

## ✅ 3. Training/Validation: Forward Pass 시 Stage 적용

**파일**: `src/archs/flow_model_vlm_film.py` (lines 373-390)

```python
def _get_vlm_film_cond(self, images: torch.Tensor, batch: dict | None = None) -> dict | None:
    # ... VLM conditioning 계산 ...
    
    # Add stage indices to vlm_cond for decoder routing
    if vlm_cond is not None and self._vlm_film_stage_indices is not None:
        vlm_cond['_vlm_film_stage_indices'] = self._vlm_film_stage_indices
```

**파일**: `src/archs/components/diffusion_unet_vlm_film.py` (lines 64-75)

```python
# Apply FiLM based on _vlm_film_stage_indices
if vlm_cond is not None and vlm_film_heads is not None:
    stage_indices = vlm_cond.get('_vlm_film_stage_indices')
    if stage_indices is not None and stage_idx in stage_indices:
        film_head_idx = stage_indices.index(stage_idx)
        print(f"[Decoder] Applying FiLM at stage_idx={stage_idx}, film_head_idx={film_head_idx}")
        x, _payload = apply_vlm_film(...)
```

**상태**: ✅ **정상**
- `self._vlm_film_stage_indices`를 `vlm_cond` dict에 담아 decoder로 전달
- Decoder는 `stage_idx`가 `stage_indices`에 포함될 때만 FiLM 적용
- Training과 Validation에서 동일한 로직 사용

---

## ✅ 4. Checkpoint 저장: Stage 메타데이터 저장

**파일**: `src/archs/flow_model_vlm_film.py` (lines 1476-1480)

```python
def on_save_checkpoint(self, checkpoint):
    """Save metadata about which stages FiLM was applied to during training."""
    super().on_save_checkpoint(checkpoint)
    if self.use_vlm_film and self._vlm_film_stage_indices is not None:
        checkpoint['vlm_film_trained_stages'] = self._vlm_film_stage_indices
```

**상태**: ✅ **정상**
- Training에서 실제 사용한 stage를 `vlm_film_trained_stages` 메타데이터로 저장
- Hparams에도 `vlm_film_decoder_stages`가 자동 저장됨

---

## ✅ 5. Checkpoint 로드: Stage 복원 (3단계 우선순위)

**파일**: `src/archs/flow_model_vlm_film.py` (lines 1482-1527)

```python
def on_load_checkpoint(self, checkpoint):
    # Priority 1: Use trained stages from checkpoint metadata (most reliable)
    trained_stages = checkpoint.get('vlm_film_trained_stages')
    if trained_stages is not None:
        self._vlm_film_stage_indices = sorted(trained_stages)
        return
    
    # Priority 2: Use hparams if available
    if hasattr(self.hparams, 'vlm_film_decoder_stages'):
        decoder_stages = self.hparams.vlm_film_decoder_stages
        if decoder_stages is not None:
            self._vlm_film_stage_indices = sorted(decoder_stages)
            return
    
    # Priority 3: Legacy checkpoint - infer from head output dimensions
    dim_to_stage = {256: 0, 192: 1, 128: 2, 64: 3}
    # ... head 출력 차원으로 stage 추론 ...
```

**상태**: ✅ **정상**
- 새 checkpoint: metadata → hparams → 차원 추론 순서로 복원
- 복원된 stage를 `self._vlm_film_stage_indices`에 설정
- Validation/Inference에서 동일한 stage 사용 보장

---

## ✅ 6. Evaluation: Checkpoint에서 Stage 검출

**파일**: `src/runner/eval_runner_vlm_film.py` (lines 212-263)

```python
def _get_vlm_film_stages_from_checkpoint(self, checkpoint_path: Path) -> list | None:
    # Priority 1: Check for trained stages metadata (new checkpoints)
    trained_stages = ckpt.get('vlm_film_trained_stages')
    if trained_stages is not None:
        return sorted(trained_stages)
    
    # Priority 2: Try hparams
    vlm_stages = hparams.get('vlm_film_decoder_stages')
    if vlm_stages is not None:
        return vlm_stages
    
    # Priority 3: Infer from head output dimensions
    # ... 256→Stage0, 192→Stage1, 128→Stage2, 64→Stage3 매핑 ...
```

**파일**: `src/runner/eval_runner_vlm_film.py` (lines 320-332)

```python
# Evaluation 시 stage 전달
vlm_stages = self._get_vlm_film_stages_from_checkpoint(checkpoint_path)
model = FlowModelVLMFiLM.load_from_checkpoint(
    str(checkpoint_path),
    vlm_film_decoder_stages=vlm_stages,  # ← Stage 명시적 전달
)
```

**상태**: ✅ **정상**
- Checkpoint에서 stage 검출 (metadata → hparams → 차원 추론)
- 검출된 stage를 `load_from_checkpoint`의 `vlm_film_decoder_stages` 인자로 전달
- 모델 로드 후 `on_load_checkpoint` 훅에서 다시 한번 검증

---

## 🎯 종합 결론

### ✅ **전체 파이프라인 일관성 확인됨**

```
Config (vlm_film_decoder_stages: [2, 3])
    ↓
Training Runner → FlowModelVLMFiLM.__init__
    ↓
self._vlm_film_stage_indices = [2, 3]  ← 여기서 설정됨
    ↓
Training Forward Pass
    ├─ _get_vlm_film_cond: vlm_cond['_vlm_film_stage_indices'] = [2, 3]
    └─ Decoder: stage_idx in [2, 3]일 때만 FiLM 적용
    ↓
Validation (동일한 로직)
    ↓
Checkpoint 저장
    ├─ vlm_film_trained_stages = [2, 3] (metadata)
    └─ hparams.vlm_film_decoder_stages = [2, 3] (hparams)
    ↓
Evaluation: Checkpoint 로드
    ├─ _get_vlm_film_stages_from_checkpoint → [2, 3]
    ├─ load_from_checkpoint(vlm_film_decoder_stages=[2, 3])
    └─ on_load_checkpoint: self._vlm_film_stage_indices = [2, 3]
    ↓
Evaluation Forward Pass (training과 동일한 로직)
    └─ stage [2, 3]에만 FiLM 적용
```

### 🔍 추가 검증 포인트

1. **Junction Gating**:
   - `apply_stages: same_as_film` 설정 시 FiLM과 동일한 stage 사용
   - Config 확인: `junction_gating_config.apply_stages`

2. **Legacy Checkpoint**:
   - 2024년 checkpoint는 metadata 없음 → hparams 또는 차원 추론 사용
   - 차원 추론 로직이 정확히 동작하는지 검증 완료 (이전 수정 사항)

3. **기본값 처리**:
   - `vlm_film_decoder_stages=None`일 때 → [0, 1] 사용 (기본값)
   - 모든 코드 경로에서 일관된 기본값 사용

### ✅ 결론

**모든 단계에서 Config의 `vlm_film_decoder_stages` 설정이 정확히 적용됩니다.**
- Training, Validation, Evaluation 모두 동일한 stage에 FiLM 적용
- Checkpoint 저장/로드 시 메타데이터와 hparams로 stage 정보 보존
- Legacy checkpoint도 차원 기반 추론으로 정확한 stage 복원
