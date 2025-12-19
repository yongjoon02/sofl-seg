"""Unit tests for the registry system.

Run with:
    cd /home/yongjun/soft-seg
    uv run pytest tests/test_registry.py -v
"""

import pytest
import sys
sys.path.insert(0, '/home/yongjun/soft-seg')

import autorootcwd


class TestRegistryBase:
    """Test base Registry class."""
    
    def test_registry_creation(self):
        """Registry 인스턴스 생성 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        assert reg._name == 'test'
        assert len(reg) == 0
    
    def test_register_with_decorator(self):
        """데코레이터로 등록 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        @reg.register(name='my_class')
        class MyClass:
            pass
        
        assert 'my_class' in reg
        assert reg.get('my_class') == MyClass
    
    def test_register_with_function_call(self):
        """함수 호출로 등록 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        class MyClass:
            pass
        
        reg.register(name='my_class', obj=MyClass)
        
        assert 'my_class' in reg
        assert reg.get('my_class') == MyClass
    
    def test_register_with_metadata(self):
        """메타데이터와 함께 등록 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        @reg.register(name='my_class', metadata={'version': '1.0', 'author': 'test'})
        class MyClass:
            pass
        
        assert 'my_class' in reg
        metadata = reg.get_metadata('my_class')
        assert metadata['version'] == '1.0'
        assert metadata['author'] == 'test'
    
    def test_register_auto_name(self):
        """이름 없이 등록 시 클래스명 사용 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        @reg.register()
        class AutoNamedClass:
            pass
        
        assert 'AutoNamedClass' in reg
    
    def test_get_nonexistent_raises(self):
        """존재하지 않는 항목 조회 시 KeyError 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        with pytest.raises(KeyError):
            reg.get('nonexistent')
    
    def test_list_with_filter(self):
        """필터링된 목록 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        reg.register(name='model_a', obj=lambda: None, metadata={'task': 'supervised'})
        reg.register(name='model_b', obj=lambda: None, metadata={'task': 'diffusion'})
        reg.register(name='model_c', obj=lambda: None, metadata={'task': 'supervised'})
        
        supervised = reg.list(task='supervised')
        assert len(supervised) == 2
        assert 'model_a' in supervised
        assert 'model_c' in supervised
        assert 'model_b' not in supervised
    
    def test_keys_values_items(self):
        """keys, values, items 메서드 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        class A: pass
        class B: pass
        
        reg.register(name='a', obj=A)
        reg.register(name='b', obj=B)
        
        assert set(reg.keys()) == {'a', 'b'}
        assert set(reg.values()) == {A, B}
        assert dict(reg.items()) == {'a': A, 'b': B}
    
    def test_iteration(self):
        """반복 테스트."""
        from src.registry.base import Registry
        
        reg = Registry('test')
        
        class A: pass
        class B: pass
        
        reg.register(name='a', obj=A)
        reg.register(name='b', obj=B)
        
        # __iter__는 name들을 반환
        names = list(reg)
        assert set(names) == {'a', 'b'}
        
        # items()로 (name, obj) 쌍 얻기
        items = dict(reg.items())
        assert items == {'a': A, 'b': B}


class TestGlobalRegistries:
    """Test global registry instances."""
    
    def test_global_registries_exist(self):
        """전역 레지스트리 존재 확인."""
        from src.registry import (
            MODEL_REGISTRY,
            DATASET_REGISTRY,
            ARCHS_REGISTRY,
            LOSS_REGISTRY,
            METRIC_REGISTRY,
        )
        
        assert MODEL_REGISTRY is not None
        assert DATASET_REGISTRY is not None
        assert ARCHS_REGISTRY is not None
        assert LOSS_REGISTRY is not None
        assert METRIC_REGISTRY is not None


class TestModelRegistry:
    """Test model registry."""
    
    def test_builtin_models_registered(self):
        """내장 모델 등록 확인."""
        from src.registry import MODEL_REGISTRY, list_models
        
        models = list_models()
        assert 'csnet' in models
        assert 'dscnet' in models
        assert 'medsegdiff' in models
        assert 'berdiff' in models
        assert 'dhariwal_concat_unet' in models
    
    def test_get_model_info(self):
        """모델 정보 조회 테스트."""
        from src.registry import get_model_info
        
        info = get_model_info('csnet')
        
        assert info.name == 'csnet'
        assert info.task == 'supervised'
        assert info.params == 8_400_196
        assert info.speed == 'fast'
        assert info.default_lr == 2e-3
        assert info.class_ref is not None
    
    def test_list_models_by_task(self):
        """태스크별 모델 목록 테스트."""
        from src.registry import list_models
        
        supervised = list_models(task='supervised')
        assert 'csnet' in supervised
        assert 'dscnet' in supervised
        
        diffusion = list_models(task='diffusion')
        assert 'medsegdiff' in diffusion
        assert 'berdiff' in diffusion
    
    def test_register_model_decorator(self):
        """모델 등록 데코레이터 테스트."""
        from src.registry import register_model, MODEL_REGISTRY, get_model_info
        
        @register_model(
            name='test_model_unit',
            task='supervised',
            params=100,
            speed='fast',
            description='Test model for unit test',
        )
        class TestModel:
            pass
        
        assert 'test_model_unit' in MODEL_REGISTRY
        
        info = get_model_info('test_model_unit')
        assert info.task == 'supervised'
        assert info.params == 100


class TestDatasetRegistry:
    """Test dataset registry."""
    
    @pytest.fixture(autouse=True)
    def setup_datasets(self):
        """데이터셋 모듈을 import하여 등록되도록 함."""
        # 데이터 모듈들을 import하면 @DATASET_REGISTRY.register가 실행됨
        import src.data.octa500
        import src.data.rossa
        import src.data.xca
    
    def test_builtin_datasets_registered(self):
        """내장 데이터셋 등록 확인."""
        from src.registry import DATASET_REGISTRY, list_datasets
        
        datasets = list_datasets()
        assert 'octa500_3m' in datasets
        assert 'octa500_6m' in datasets
        assert 'rossa' in datasets
        assert 'xca' in datasets
    
    def test_get_dataset_info(self):
        """데이터셋 정보 조회 테스트."""
        from src.registry import get_dataset_info
        
        info = get_dataset_info('xca')
        
        assert info.name == 'xca'
        assert info.modality == 'XCA'
        assert info.resolution == (512, 512)
        assert info.num_train == 155
        assert info.num_val == 20
        assert info.num_test == 46
        assert info.default_crop_size == 320
        assert info.class_ref is not None
    
    def test_list_datasets_by_modality(self):
        """모달리티별 데이터셋 목록 테스트."""
        from src.registry import list_datasets
        
        octa = list_datasets(modality='OCTA')
        assert 'octa500_3m' in octa
        assert 'octa500_6m' in octa
        assert 'rossa' in octa
        
        xca = list_datasets(modality='XCA')
        assert 'xca' in xca
    
    def test_register_dataset_decorator(self):
        """데이터셋 등록 데코레이터 테스트."""
        from src.registry import register_dataset, DATASET_REGISTRY, get_dataset_info
        
        @register_dataset(
            name='test_dataset_unit',
            modality='TEST',
            resolution=(256, 256),
            num_train=10,
            num_val=2,
            num_test=3,
            description='Test dataset for unit test',
        )
        class TestDataset:
            pass
        
        assert 'test_dataset_unit' in DATASET_REGISTRY
        
        info = get_dataset_info('test_dataset_unit')
        assert info.modality == 'TEST'
        assert info.resolution == (256, 256)


class TestArchRegistry:
    """Test architecture registry."""
    
    def test_register_arch_decorator(self):
        """아키텍처 등록 데코레이터 테스트."""
        from src.registry import register_arch, ARCHS_REGISTRY, get_arch_info
        
        @register_arch(
            name='test_arch_unit',
            in_channels=1,
            out_channels=2,
            description='Test architecture',
        )
        class TestArch:
            pass
        
        assert 'test_arch_unit' in ARCHS_REGISTRY
        
        info = get_arch_info('test_arch_unit')
        assert info.in_channels == 1
        assert info.out_channels == 2


class TestLossRegistry:
    """Test loss registry."""
    
    def test_register_loss_decorator(self):
        """Loss 등록 데코레이터 테스트."""
        from src.registry import register_loss, LOSS_REGISTRY, get_loss_info
        
        @register_loss(
            name='test_loss_unit',
            description='Test loss function',
            supports_soft_labels=True,
        )
        class TestLoss:
            pass
        
        assert 'test_loss_unit' in LOSS_REGISTRY
        
        info = get_loss_info('test_loss_unit')
        assert info.supports_soft_labels == True


class TestMetricRegistry:
    """Test metric registry."""
    
    def test_register_metric_decorator(self):
        """Metric 등록 데코레이터 테스트."""
        from src.registry import register_metric, METRIC_REGISTRY, get_metric_info
        
        @register_metric(
            name='test_metric_unit',
            description='Test metric',
            higher_is_better=False,
            range=(0.0, 100.0),
        )
        class TestMetric:
            pass
        
        assert 'test_metric_unit' in METRIC_REGISTRY
        
        info = get_metric_info('test_metric_unit')
        assert info.higher_is_better == False
        assert info.range == (0.0, 100.0)


class TestBackwardCompatibility:
    """Test backward compatibility with old imports."""
    
    def test_old_import_path(self):
        """기존 import 경로 호환성 테스트 (deprecated - kept for reference)."""
        # Note: src.utils.registry is deprecated, all code should use src.registry
        # This test verifies that the new import path works correctly
        from src.registry import (
            ARCHS_REGISTRY,
            DATASET_REGISTRY,
            MODEL_REGISTRY,
            LOSS_REGISTRY,
            METRIC_REGISTRY,
        )
        
        # Verify they are Registry instances
        from src.registry import Registry
        assert isinstance(ARCHS_REGISTRY, Registry)
        assert isinstance(DATASET_REGISTRY, Registry)
        assert isinstance(MODEL_REGISTRY, Registry)


class TestModelClassInstantiation:
    """Test that registered classes can be instantiated."""
    
    def test_csnet_instantiation(self):
        """CSNet 인스턴스 생성 테스트."""
        from src.registry import MODEL_REGISTRY
        
        CSNet = MODEL_REGISTRY.get('csnet')
        model = CSNet(in_channels=1, num_classes=2)
        
        assert model is not None
    
    def test_dataset_instantiation(self):
        """DataModule 인스턴스 생성 테스트."""
        # 먼저 데이터 모듈 import하여 등록
        import src.data.xca
        from src.registry import DATASET_REGISTRY
        
        XCADataModule = DATASET_REGISTRY.get('xca')
        
        # Just check class exists, don't instantiate (needs data files)
        assert XCADataModule is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

