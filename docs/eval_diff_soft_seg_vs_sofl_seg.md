# DFM+VLM Eval: soft-seg vs sofl-seg 차이

## 요약

| 항목 | soft-seg (`/home/yongjun/soft-seg`) | sofl-seg (`/home/yongjun/sofl-seg`) |
|------|-------------------------------------|-------------------------------------|
| VLM 전용 eval | **없음** | **있음** (`eval_runner_vlm_film.py` + `evaluate_vlm_film.py`) |
| Flow 체크포인트 로드 | 항상 `FlowModel` (또는 `FlowSoft2HardModel`) | 일반: `FlowModel` / VLM: `FlowModelVLMFiLM` 분기 |
| VLM 체크포인트 감지 | 없음 | `_is_vlm_checkpoint()` |
| VLM Film stage 로드 | 해당 없음 | `_get_vlm_film_stages_from_checkpoint()` → `vlm_film_decoder_stages` 전달 |
| FlowModelVLMFiLM | **없음** (arch에 해당 파일 없음) | **있음** (`flow_model_vlm_film.py`) |

---

## 1. soft-seg (`/home/yongjun/soft-seg`)

- **Eval 진입점**: `scripts/evaluate.py` → `EvalRunner` (`src/runner/eval_runner.py`) 만 사용.
- **Flow 로딩** (`eval_runner.py` 231–237행):
  - `model_info.task == 'flow'` 이면
  - `medsegdiff_flow_soft2hard`, `medsegdiff_flow_multitask` → `FlowSoft2HardModel`
  - 그 외 → **항상 `FlowModel`**
- **VLM 관련**:
  - `FlowModelVLMFiLM` 클래스 없음 (`src/archs/` 에 `flow_model_vlm_film.py` 없음).
  - `eval_runner_vlm_film.py`, `evaluate_vlm_film.py` 없음.
  - VLM으로 학습한 체크포인트도 **그대로 `FlowModel`로 로드** → VLM head/conditioner 없음, stage 수 불일치 등으로 예측이 깨질 수 있음.

---

## 2. sofl-seg (`/home/yongjun/sofl-seg`) – DFM+VLM 전용 경로

- **Eval 진입점**: `scripts/evaluate_vlm_film.py` → `EvalRunnerVLMFiLM` (`src/runner/eval_runner_vlm_film.py`).
- **Flow 로딩** (`eval_runner_vlm_film.py`):
  - `_is_vlm_checkpoint(checkpoint_path)` 로 VLM 체크포인트 여부 판별.
  - **VLM이면**: `FlowModelVLMFiLM.load_from_checkpoint(..., vlm_film_decoder_stages=vlm_stages)`  
    - `vlm_stages` = `_get_vlm_film_stages_from_checkpoint()` 로 state_dict에서 head 개수 감지 (예: 4개면 `[0,1,2,3]`).
  - **VLM 아니면**: `FlowModel.load_from_checkpoint(...)`.
- **추가 옵션**: `save_intermediate`, `intermediate_t`, `intermediate_steps`, `intermediate_max_samples` 등으로 중간 step 저장 가능.

---

## 3. 왜 sofl-seg에서만 DFM+VLM이 “제대로” 도는가

- VLM 학습 시 사용한 모델은 **`FlowModelVLMFiLM`** (VLM conditioner + decoder stage별 FiLM head).
- soft-seg는 이 클래스가 없고, **모든 flow를 `FlowModel`로만 로드**하므로:
  - VLM 체크포인트를 넣어도 `FlowModel`로 로드됨.
  - `vlm_film_heads`, `vlm_film_conditioner` 등이 없어서 weight 불일치/누락 발생 → 예측이 틀어짐.
- sofl-seg는 **VLM 체크포인트일 때만 `FlowModelVLMFiLM`** 를 쓰고, **체크포인트 state_dict 기준으로 stage 수**를 맞춰서 로드하기 때문에 DFM+VLM 예측이 올바르게 동작함.

---

## 4. 정리

- **다른 부분**: soft-seg에는 **DFM+VLM 전용 eval 코드가 없음** (전용 runner, 전용 스크립트, `FlowModelVLMFiLM` 자체가 없음).  
  sofl-seg에는 **DFM+VLM 전용** `EvalRunnerVLMFiLM` + `evaluate_vlm_film.py` 가 있고, VLM 체크포인트 감지 + stage 자동 감지로 `FlowModelVLMFiLM`를 올바르게 로드함.
- **DFM+VLM 체크포인트를 제대로 eval 하려면** sofl-seg처럼 **VLM 전용 eval 경로**에서 `FlowModelVLMFiLM` 로 로드하는 구조가 필요함.
