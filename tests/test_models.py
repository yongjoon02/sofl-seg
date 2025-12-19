"""모델 아키텍처 테스트 (pytest)"""
import pytest
import torch
import sys
sys.path.insert(0, '/home/yongjun/soft-seg')

from src.archs.components import CSNet, DSCNet


class TestModelArchitectures:
    """Supervised 모델 아키텍처 테스트"""
    
    @pytest.fixture
    def input_tensor(self):
        """테스트용 입력 텐서"""
        return torch.randn(2, 1, 224, 224)
    
    @pytest.mark.parametrize("ModelClass,model_name", [
        (CSNet, 'csnet'),
        (DSCNet, 'dscnet'),
    ])
    def test_model_initialization(self, ModelClass, model_name):
        """모델이 정상적으로 초기화되는지 확인"""
        model = ModelClass(in_channels=1, num_classes=2)
        assert model is not None
        print(f"✓ {model_name} initialized")
    
    @pytest.mark.parametrize("ModelClass,model_name", [
        (CSNet, 'csnet'),
        (DSCNet, 'dscnet'),
    ])
    def test_model_forward(self, ModelClass, model_name, input_tensor):
        """Forward pass가 정상 작동하는지 확인"""
        model = ModelClass(in_channels=1, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # 출력 shape 확인
        if isinstance(output, dict):
            assert 'out' in output or 'main' in output
        else:
            assert output.shape[0] == input_tensor.shape[0]  # batch size
            assert output.shape[1] == 2  # num_classes
            assert output.shape[2] == input_tensor.shape[2]  # height
            assert output.shape[3] == input_tensor.shape[3]  # width
        
        print(f"✓ {model_name} forward pass successful")
    
    @pytest.mark.parametrize("ModelClass", [
        CSNet, DSCNet
    ])
    def test_model_parameters_count(self, ModelClass):
        """모델 파라미터 수 확인"""
        model = ModelClass(in_channels=1, num_classes=2)
        
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        print(f"✓ {ModelClass.__name__}: {num_params:,} parameters")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
