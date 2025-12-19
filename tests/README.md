# Tests

Modern pytest-based unit tests for the soft-seg project.

## Running Tests

```bash
# Install pytest
uv pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_datasets.py -v

# Run specific test
pytest tests/test_models.py::TestModelArchitectures::test_model_forward -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests for specific dataset
pytest tests/test_datasets.py -v -k "octa500_3m"

# Run tests for specific model
pytest tests/test_models.py -v -k "csnet"
```

## Test Structure

### 🗂️ test_datasets.py
**모든 데이터셋 통합 테스트** (parametrized)
- ✅ Registry 등록 확인
- ✅ DataModule 초기화
- ✅ DataLoader 생성
- ✅ Batch 로딩 및 검증
- 📊 Datasets: OCTA500_3M, OCTA500_6M, ROSSA, XCA

### 🏗️ test_models.py
**모든 모델 아키텍처 테스트** (parametrized)
- ✅ 모델 초기화
- ✅ Forward pass 검증
- ✅ Output shape 확인
- ✅ Parameter count
- 🤖 Models: CENet, CSNet, AACAUNet, UNet3Plus, VesselNet, TransUNet, DSCNet

### 🔗 test_integration.py
**데이터셋 + 모델 통합 테스트**
- ✅ Supervised 모델 + 데이터셋 조합
- ✅ Diffusion 모델 초기화
- ✅ End-to-end forward pass

## Advantages

✨ **Pytest 사용**:
- Parametrized tests (중복 제거)
- Automatic test discovery
- Rich assertion messages
- Fixtures for shared setup

✨ **효율적 구조**:
- 하나의 테스트로 모든 데이터셋/모델 검증
- 실패 시 정확한 위치 파악
- CI/CD 통합 용이

## Test Coverage

✅ **4개 데이터셋**: OCTA500_3M, OCTA500_6M, ROSSA, XCA  
✅ **7개 모델**: CENet, CSNet, AACAUNet, UNet3Plus, VesselNet, TransUNet, DSCNet  
✅ **통합 테스트**: 모든 조합 자동 검증

**총 테스트 수**: 50개 이상 (parametrized)
