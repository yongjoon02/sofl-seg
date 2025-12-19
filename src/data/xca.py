"""XCA (X-ray Coronary Angiography) 데이터셋 모듈

XCA 데이터는 관상동맥 조영술 이미지의 혈관 분할 데이터셋입니다.
X-ray 이미지는 ScaleIntensityd로 정규화하여 [-1, 1] 범위로 변환합니다.
"""
import math
from typing import Optional

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambda,
    RandAdjustContrastd,
    RandRotated,
    ScaleIntensityd,
)

from src.data.base_dataset import BaseOCTDataModule, BaseOCTDataset
from src.registry import DATASET_REGISTRY


class XCADataset(BaseOCTDataset):
    """XCA 데이터셋 (BaseOCTDataset 상속)
    
    데이터 구조:
        root/
            image/  - 원본 X-ray 이미지 (grayscale PNG, 512x512)
            label/  - 혈관 분할 마스크 (grayscale PNG, 512x512)
            label_smooth/   - Label smoothing soft labels (optional)
            label_gaussian/ - Gaussian boundary soft labels (optional)
            label_sauna/    - SAUNA soft labels (optional)
    
    X-ray 특화 처리:
    - RGB/RGBA → Grayscale 변환 (첫 번째 채널만 사용)
    - RandRotated: X-ray 특성에 맞는 작은 회전 (±7.5도)
    - RandAdjustContrastd: X-ray 대비 조정
    
    Args:
        label_subdir: Label subdirectory name (default: 'label')
                     Options: 'label', 'label_smooth', 'label_gaussian', 'label_sauna'
    """
    
    def __init__(
        self, 
        path: str, 
        augmentation: bool = False, 
        crop_size: int = 128,
        num_samples_per_image: int = 1, 
        label_subdir: str = 'label',
        use_sauna_transform: bool = False,  # SAUNA 동적 변환 사용 여부
    ) -> None:
        self.label_subdir = label_subdir
        self.use_sauna_transform = use_sauna_transform  # SAUNA 변환 플래그
        super().__init__(path, augmentation, crop_size, num_samples_per_image)

    def get_data_fields(self) -> list[str]:
        """
        XCA는 image와 label만 사용.
        
        SAUNA 동적 변환을 사용하는 경우에만 hard label도 함께 로드.
        """
        fields = ['image', self.label_subdir]
        # 항상 hard label을 추가로 로드해 메트릭/로그에서 안전하게 사용
        if 'label' not in fields:
            fields.append('label')
        return fields

    def _create_transforms(self):
        """X-ray 특화 transform 생성 (Base 오버라이드)"""
        # Base의 기본 transform 먼저 생성 (augmentation_transforms 포함)
        super()._create_transforms()

        keys = self.fields
        label_key = self.label_subdir  # 동적 label key

        # RGB→Grayscale 변환 추가 (X-ray 특화)
        def rgb_to_gray(d):
            """Convert RGB to Grayscale by taking first channel."""
            result = {**d}
            result["image"] = d["image"][:1] if hasattr(d["image"], "shape") and d["image"].shape[0] > 1 else d["image"]
            # soft label 채널
            result[label_key] = d[label_key][:1] if hasattr(d[label_key], "shape") and d[label_key].shape[0] > 1 else d[label_key]
            # hard label 채널도 항상 1채널 보장
            if "label" in d:
                result["label"] = d["label"][:1] if hasattr(d["label"], "shape") and d["label"].shape[0] > 1 else d["label"]
            return result

        # Soft label 정규화 함수 (단순 /255, min-max가 아님)
        def normalize_soft_label(d):
            """Normalize soft label by dividing by 255 (preserve actual values)."""
            result = {**d}
            if label_key in d:
                label_data = d[label_key]
                # Tensor 또는 numpy array 모두 처리
                if isinstance(label_data, torch.Tensor):
                    # 이미 tensor인 경우 (EnsureChannelFirstd 이후)
                    # 최대값이 1보다 크면 255로 나누기 (uint8 범위)
                    if label_data.max() > 1.0:
                        result[label_key] = label_data.float() / 255.0
                    else:
                        # 이미 정규화된 경우 그대로 사용
                        result[label_key] = label_data.float()
                else:
                    # numpy array인 경우
                    import numpy as np
                    label_data = np.asarray(label_data)
                    if label_data.max() > 1.0:
                        result[label_key] = torch.from_numpy(label_data).float() / 255.0
                    else:
                        result[label_key] = torch.from_numpy(label_data).float()
            return result

        # Default transforms에 RGB→Gray 추가 + hard/soft label 처리
        default_transforms = [
            EnsureChannelFirstd(keys=keys),
            rgb_to_gray,  # RGB→Grayscale
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]
        if "label" in keys:
            default_transforms.append(ScaleIntensityd(keys="label", minv=0.0, maxv=1.0))
        default_transforms.append(normalize_soft_label)  # Soft label: 단순 /255 (min-max 아님)

        self.default_transforms = Compose(default_transforms)

        # X-ray 특화 augmentation
        xray_augments = [
            RandRotated(keys=keys, range_x=(math.pi/24, math.pi/24),
                       range_y=(math.pi/24, math.pi/24), prob=0.25),
            RandAdjustContrastd(keys="image", prob=0.25, gamma=(0.9, 1.1)),
        ]

        # Augmentation transforms 재정의 (label_key 동적 설정)
        from monai.transforms import RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandSpatialCropd
        
        if self.num_samples_per_image > 1:
            # RandCropByPosNegLabeld는 내부적으로 threshold를 사용하여 pos/neg 영역 찾기
            # Soft label을 그대로 사용하되, cropping을 위한 mask 생성용으로만 binarization 사용
            self.augmentation_transforms = Compose([
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandRotate90d(keys=keys, prob=0.5, max_k=3),
            ] + xray_augments + [
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key=label_key,  # 동적 label key 사용!
                    spatial_size=(self.crop_size, self.crop_size),
                    pos=1,
                    neg=1,
                    num_samples=self.num_samples_per_image,
                    # RandCropByPosNegLabeld는 내부적으로 label > 0 체크로 pos/neg 영역 찾기
                    # Soft label 값은 그대로 유지됨
                ),
            ])
        else:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandRotate90d(keys=keys, prob=0.5, max_k=3),
            ] + xray_augments + [
                RandSpatialCropd(
                    keys=keys,
                    roi_size=(self.crop_size, self.crop_size),
                    random_size=False
                ),
            ])

    def to_geometry(self, label: torch.Tensor) -> torch.Tensor:
        """
        Convert binary label to geometry (soft label) for flow matching.
        
        Args:
            label: Binary label tensor (C, H, W) or (B, C, H, W) with values in {0, 1}
            
        Returns:
            geometry: Geometry map tensor (same shape as label) with values in [0, 1]
                - If use_sauna_transform=True: SAUNA geometry map normalized to [0, 1]
                - If use_sauna_transform=False: Simple float conversion (identity transform)
        """
        if self.use_sauna_transform:
            # SAUNA 변환 사용
            from src.data.transforms.sauna import to_geometry as sauna_to_geometry
            
            # Ensure 4D tensor (B, C, H, W)
            was_3d = label.dim() == 3
            if was_3d:
                label = label.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            
            # Ensure channel dimension is 1
            if label.size(1) != 1:
                label = label[:, 0:1, :, :]  # Take first channel
            
            # Convert to SAUNA geometry map (returns [-1, 1] range)
            geometry = sauna_to_geometry(
                label,
                use_thickness=True,
                target_c_label="h",
                kernel_size=None,
                kernel_ratio=1.0
            )
            
            # Normalize SAUNA output from [-1, 1] to [0, 1] for consistent threshold
            # This allows using threshold=0.5 in validation/test steps
            geometry = (geometry + 1.0) / 2.0
            
            # Remove batch dimension if input was 3D
            if was_3d:
                geometry = geometry.squeeze(0)
            
            return geometry
        else:
            # 기존 동작: 단순 float 변환 (identity transform)
            return label.float()
    
    def __getitem__(self, index):
        """
        Get a sample with X-ray specific post-processing.
        
        Args:
            index: Sample index
            
        Returns:
            dict: Dictionary containing:
                - image: Input image (C, H, W) normalized to [-1, 1]
                - label: Binary label (C, H, W) for metrics calculation
                - geometry: Soft label (C, H, W) for flow matching training
                - name: Sample filename
                - coordinate: Coordinate tensor (for FlowCoordModel)
        """
        # Base class의 __getitem__ 호출
        data = super().__getitem__(index)

        # X-ray 특화 후처리: 이미지 clamp (augmentation 후 범위 벗어날 수 있음)
        if self.augmentation:
            data["image"] = torch.clamp(data["image"], -1.0, 1.0)

        # Flow matching을 위한 geometry 생성
        # - label: binary (0/1) - metrics 계산용 (항상 hard label)
        # - geometry: soft label - flow matching 학습용
        if 'label' in data:
            if self.use_sauna_transform:
                # SAUNA 동적 변환 사용: hard label에서 SAUNA로 변환
                # 'label' 키는 항상 hard label이어야 함 (get_data_fields에서 로드)
                hard_label = data['label']
                
                # Hard label에서 SAUNA geometry 생성
                data['geometry'] = self.to_geometry(hard_label)
            else:
                # 기존 동작: label_subdir의 값을 geometry로 사용
                if self.label_subdir != 'label' and self.label_subdir in data:
                    # label_subdir이 'label_sauna' 등인 경우, 해당 값을 geometry로 사용
                    data['geometry'] = self.to_geometry(data[self.label_subdir])
                    # label은 hard label로 유지 (metrics 계산용)
                else:
                    # label_subdir == 'label'인 경우, label을 그대로 geometry로 사용
                    data['geometry'] = self.to_geometry(data['label'])
            
            # Debug: Check if geometry is actually soft label (only log first sample to avoid spam)
            if self.use_sauna_transform and not hasattr(self, '_geometry_check_logged'):
                geom = data['geometry']
                geom_min, geom_max = geom.min().item(), geom.max().item()
                geom_unique = torch.unique(geom).numel()
                if geom_unique <= 2 and geom_min in [0.0, 1.0] and geom_max in [0.0, 1.0]:
                    print(f"⚠️ WARNING: use_sauna_transform=True but geometry appears binary "
                          f"(unique: {geom_unique}, range: [{geom_min:.3f}, {geom_max:.3f}])")
                else:
                    print(f"✅ use_sauna_transform=True: geometry is SAUNA soft label "
                          f"(unique: {geom_unique}, range: [{geom_min:.3f}, {geom_max:.3f}])")
                self._geometry_check_logged = True

        return data


class XCADataModule(BaseOCTDataModule):
    """XCA 데이터 모듈 (BaseOCTDataModule 상속)
    
    Usage:
        datamodule = XCADataModule(
            train_dir='data/xca_full/train',
            val_dir='data/xca_full/val',
            test_dir='data/xca_full/test',
            crop_size=320,
            train_bs=8,
            num_samples_per_image=1,
            label_subdir='label_sauna',  # soft label 사용
        )
    """

    dataset_class = XCADataset

    def __init__(
        self,
        train_dir: str = 'data/xca_full/train',
        val_dir: str = 'data/xca_full/val',
        test_dir: Optional[str] = 'data/xca_full/test',
        crop_size: int = 320,
        train_bs: int = 8,
        num_samples_per_image: int = 1,
        label_subdir: str = 'label',
        use_sauna_transform: bool = False,  # SAUNA 동적 변환 사용 여부
    ):
        """XCA 데이터 모듈 초기화
        
        Args:
            train_dir: 학습 데이터 디렉토리
            val_dir: 검증 데이터 디렉토리
            test_dir: 테스트 데이터 디렉토리 (선택)
            crop_size: 크롭 크기 (default: 320, 원본 512×512의 62.5%)
            train_bs: 학습 배치 크기
            num_samples_per_image: 이미지당 크롭 샘플 수
            label_subdir: Label 서브디렉토리 ('label', 'label_smooth', 'label_gaussian', 'label_sauna')
            use_sauna_transform: True이면 hard label에서 SAUNA로 동적 변환 (기본값: False, 기존 동작 유지)
        """
        self.label_subdir = label_subdir
        self.use_sauna_transform = use_sauna_transform
        super().__init__(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            name='xca'
        )

    def create_train_dataset(self):
        """Create training dataset from single directory"""
        return self.dataset_class(
            self.train_dir,
            augmentation=True,
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image,
            label_subdir=self.label_subdir,
            use_sauna_transform=self.use_sauna_transform,
        )
    
    def create_val_dataset(self):
        """Create validation dataset"""
        return self.dataset_class(
            self.val_dir,
            augmentation=False,
            crop_size=self.crop_size,
            num_samples_per_image=1,
            label_subdir=self.label_subdir,
            use_sauna_transform=self.use_sauna_transform,
        )
    
    def create_test_dataset(self):
        """Create test dataset"""
        if self.test_dir is None:
            return None
        return self.dataset_class(
            self.test_dir,
            augmentation=False,
            crop_size=self.crop_size,
            num_samples_per_image=1,
            label_subdir=self.label_subdir,
            use_sauna_transform=self.use_sauna_transform,
        )


@DATASET_REGISTRY.register(name='xca')
class XCA_DataModule(XCADataModule):
    """Registry에 등록된 XCA 데이터 모듈 (기본 파라미터)"""
    def __init__(
        self,
        train_dir: str = 'data/xca_full/train',
        val_dir: str = 'data/xca_full/val',
        test_dir: Optional[str] = 'data/xca_full/test',
        crop_size: int = 320,
        train_bs: int = 8,
        num_samples_per_image: int = 1,
        label_subdir: str = 'label',
        use_sauna_transform: bool = False,
    ):
        super().__init__(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            label_subdir=label_subdir,
            use_sauna_transform=use_sauna_transform,
        )


if __name__ == '__main__':
    from src.utils.visualize_dataloader import visualize_dataset

    print("=" * 60)
    print("XCA 데이터셋 테스트 (BaseOCTDataset 상속)")
    print("=" * 60)

    # Registry에서 가져오기
    dm = DATASET_REGISTRY.get('xca')()
    dm.setup()

    # 데이터 개수 확인
    print("\n📊 데이터 개수:")
    print(f"   Train: {len(dm.train_dataset)} 샘플")
    print(f"   Val:   {len(dm.val_dataset)} 샘플")
    if dm.test_dataset:
        print(f"   Test:  {len(dm.test_dataset)} 샘플")

    # 샘플 데이터 확인
    train_sample = dm.train_dataset[0]
    print("\n📦 샘플 데이터 shape:")
    print(f"   Image: {train_sample['image'].shape} (range: {train_sample['image'].min():.2f} ~ {train_sample['image'].max():.2f})")
    print(f"   Label: {train_sample['label'].shape} (range: {train_sample['label'].min():.2f} ~ {train_sample['label'].max():.2f})")

    # 시각화
    visualize_dataset(dm.train_dataloader(), "xca_train", num_samples=10)
    visualize_dataset(dm.val_dataloader(), "xca_val")
    if dm.test_dataset:
        visualize_dataset(dm.test_dataloader(), "xca_test")

    print("\n✅ XCA 데이터셋 테스트 완료!")
