"""통합 테스트 (pytest)"""
import pytest
import torch
import sys
sys.path.insert(0, '/home/yongjun/soft-seg')

from src.registry.datasets import DATASET_REGISTRY
from src.registry.models import get_model_info
from src.archs.supervised_model import SupervisedModel
from src.archs.diffusion_model import DiffusionModel


class TestIntegration:
    """데이터셋 + 모델 통합 테스트"""
    
    @pytest.mark.parametrize("model_name", [
        'csnet',
        'dscnet',
    ])
    @pytest.mark.parametrize("dataset_name", [
        'octa500_3m',
        'xca',
    ])
    def test_supervised_model_with_dataset(self, model_name, dataset_name):
        """Supervised 모델 + 데이터셋 통합 테스트"""
        # DataModule 생성
        dataset_info = DATASET_REGISTRY.get(dataset_name)
        DataModuleClass = dataset_info.class_ref
        datamodule = DataModuleClass()
        datamodule.setup()
        
        # 배치 가져오기
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        # 모델 생성
        model = SupervisedModel(
            arch_name=model_name,
            in_channels=1,
            num_classes=2,
            learning_rate=0.001
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch['image'])
        
        assert output.shape[0] == batch['image'].shape[0]
        assert output.shape[1] == 2
        print(f"✓ {model_name} + {dataset_name} integration successful")
    
    @pytest.mark.parametrize("model_name", [
        'medsegdiff',
        'berdiff',
    ])
    def test_diffusion_model_initialization(self, model_name):
        """Diffusion 모델 초기화 테스트"""
        model = DiffusionModel(
            arch_name=model_name,
            image_size=224,
            dim=32,
            timesteps=50,
            learning_rate=0.0002,
            num_classes=2
        )
        assert model is not None
        print(f"✓ {model_name} initialized")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
