"""데이터셋 로딩 테스트 (pytest)"""
import pytest
import sys
sys.path.insert(0, '/home/yongjun/soft-seg')

from src.registry.datasets import get_dataset_info, DATASET_REGISTRY


class TestDatasets:
    """모든 데이터셋에 대한 통합 테스트"""
    
    @pytest.mark.parametrize("dataset_name", [
        'octa500_3m',
        'octa500_6m', 
        'rossa',
        'xca'
    ])
    def test_dataset_registry(self, dataset_name):
        """레지스트리에 데이터셋이 등록되어 있는지 확인"""
        info = get_dataset_info(dataset_name)
        assert info is not None
        assert info.name == dataset_name
        assert info.num_train > 0
        assert info.num_val > 0
        assert info.num_test > 0
    
    @pytest.mark.parametrize("dataset_name", [
        'octa500_3m',
        'octa500_6m',
        'rossa',
        'xca'
    ])
    def test_datamodule_initialization(self, dataset_name):
        """DataModule이 정상적으로 초기화되는지 확인"""
        dataset_info = DATASET_REGISTRY.get(dataset_name)
        assert dataset_info is not None
        
        DataModuleClass = dataset_info.class_ref
        datamodule = DataModuleClass()
        assert datamodule is not None
        
        # setup 호출
        datamodule.setup()
        
    @pytest.mark.parametrize("dataset_name", [
        'octa500_3m',
        'octa500_6m',
        'rossa',
        'xca'
    ])
    def test_dataloader_creation(self, dataset_name):
        """DataLoader가 정상적으로 생성되는지 확인"""
        dataset_info = DATASET_REGISTRY.get(dataset_name)
        DataModuleClass = dataset_info.class_ref
        datamodule = DataModuleClass()
        datamodule.setup()
        
        # DataLoader 생성
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
    @pytest.mark.parametrize("dataset_name", [
        'octa500_3m',
        'octa500_6m',
        'rossa',
        'xca'
    ])
    def test_batch_loading(self, dataset_name):
        """배치가 정상적으로 로드되는지 확인"""
        dataset_info = DATASET_REGISTRY.get(dataset_name)
        DataModuleClass = dataset_info.class_ref
        datamodule = DataModuleClass()
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        # 기본 키 확인
        assert 'image' in batch
        assert 'label' in batch
        assert 'name' in batch
        
        # Shape 확인
        assert batch['image'].ndim == 4  # [B, C, H, W]
        assert batch['label'].ndim == 4
        assert batch['image'].shape[0] > 0  # batch size > 0
        assert batch['image'].shape[1] == 1  # grayscale
        
        # Value range 확인
        assert batch['image'].max() <= 1.0
        assert batch['image'].min() >= -1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
