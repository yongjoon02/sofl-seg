"""Base classes for OCT/OCTA/Retinal vessel segmentation datasets.

This module provides base classes that eliminate code duplication between
different dataset implementations (OCTA500, ROSSA, etc.).
"""
import os
from abc import ABC, abstractmethod

import lightning as L
import torch
from monai.data import PILReader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    ScaleIntensityd,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Field-specific scaling configuration
FIELD_SCALE_CONFIG = {
    "image": (-1.0, 1.0),
    "label": (0.0, 1.0),
}


class BaseOCTDataset(Dataset, ABC):
    """
    Base class for OCT/OCTA/Retinal vessel segmentation datasets.
    
    This class handles common functionality:
    - File loading and validation
    - Transform setup (augmentation and normalization)
    - Indexing and sample generation
    
    Subclasses must implement:
    - get_data_fields(): List of data fields to load (e.g., ['image', 'label', 'label_prob', 'label_sauna'])
    
    Example:
        class OCTADataset(BaseOCTDataset):
            def get_data_fields(self):
                return ['image', 'label', 'label_prob', 'label_sauna']
    """

    @abstractmethod
    def get_data_fields(self) -> list[str]:
        """
        Return list of data fields to load.
        
        Returns:
            list[str]: Field names (e.g., ['image', 'label', 'label_prob', 'label_sauna'])
                      These correspond to subdirectories in the dataset path.
        """
        pass

    def __init__(self, path: str, augmentation: bool = False, crop_size: int = 128,
                 num_samples_per_image: int = 1) -> None:
        """
        Args:
            path: Dataset split path (e.g., data/OCTA500_3M/train)
            augmentation: Whether to apply data augmentation (True for training only)
            crop_size: Random crop size (roi_size)
            num_samples_per_image: Number of samples per image (default: 1)
        """
        super().__init__()
        self.path = path
        self.augmentation = augmentation
        self.crop_size = crop_size
        self.num_samples_per_image = num_samples_per_image

        # Get fields from subclass
        self.fields = self.get_data_fields()

        # Validate that 'image' field exists
        if 'image' not in self.fields:
            raise ValueError("'image' must be included in data fields")

        # Dynamically create directory paths for each field
        for field in self.fields:
            field_dir = os.path.join(path, field)
            setattr(self, f"{field}_dir", field_dir)

        # List all image files
        self.image_dir = os.path.join(path, "image")
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")

        self.image_files = sorted(os.listdir(self.image_dir))

        # Validate and collect valid data samples
        self.data = []
        for file in self.image_files:
            # Build paths for all fields
            file_paths = {}
            for field in self.fields:
                field_dir = getattr(self, f"{field}_dir")
                file_paths[field] = os.path.join(field_dir, file)

            # Check if all required files exist
            if all(os.path.exists(p) for p in file_paths.values()):
                file_paths["name"] = f"{os.path.basename(path)}/image/{file}"
                self.data.append(file_paths)
            else:
                missing = [p for p in file_paths.values() if not os.path.exists(p)]
                print(f"Warning: Missing files for {file}: {missing}")

        if len(self.data) == 0:
            fields_str = ", ".join(self.fields)
            raise ValueError(
                f"No valid data found in {path}. "
                f"Check if {fields_str} directories exist and contain matching files."
            )

        # Setup image loader
        self.image_loader = LoadImage(reader=PILReader(), image_only=True)

        # Create transforms
        self._create_transforms()

    def _create_transforms(self):
        """Create default and augmentation transforms based on data fields."""
        keys = self.fields

        # Default transforms (normalization)
        scale_transforms = [EnsureChannelFirstd(keys=keys)]

        # Group fields by their scale configuration
        for field in keys:
            if field in FIELD_SCALE_CONFIG:
                minv, maxv = FIELD_SCALE_CONFIG[field]
                # Check if we can group with other fields with same scale
                same_scale_fields = [
                    f for f in keys
                    if f in FIELD_SCALE_CONFIG and FIELD_SCALE_CONFIG[f] == (minv, maxv)
                ]
                # Add transform for this group (will be deduplicated by Compose)
                scale_transforms.append(
                    ScaleIntensityd(keys=same_scale_fields, minv=minv, maxv=maxv)
                )

        # Remove duplicate transforms by using a set to track added scale configs
        seen_scales = set()
        unique_scale_transforms = [scale_transforms[0]]  # Keep EnsureChannelFirstd

        for transform in scale_transforms[1:]:
            # Create a hashable key from the transform's keys
            transform_keys = tuple(sorted(transform.keys))
            if transform_keys not in seen_scales:
                seen_scales.add(transform_keys)
                unique_scale_transforms.append(transform)

        self.default_transforms = Compose(unique_scale_transforms)

        # Augmentation transforms
        if self.num_samples_per_image > 1:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandRotate90d(keys=keys, prob=0.5, max_k=3),
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="label",
                    spatial_size=(self.crop_size, self.crop_size),
                    pos=1,
                    neg=1,
                    num_samples=self.num_samples_per_image,
                ),
            ])
        else:
            self.augmentation_transforms = Compose([
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
                RandRotate90d(keys=keys, prob=0.5, max_k=3),
                RandSpatialCropd(
                    keys=keys,
                    roi_size=(self.crop_size, self.crop_size),
                    random_size=False
                ),
            ])

    def __len__(self):
        """
        Return total number of samples in the dataset.
        
        If num_samples_per_image > 1, returns len(data) * num_samples_per_image
        """
        if self.augmentation and self.num_samples_per_image > 1:
            return len(self.data) * self.num_samples_per_image
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            dict: Dictionary containing image, labels, and metadata
        """
        # Map index to actual data index when using multiple samples per image
        if self.augmentation and self.num_samples_per_image > 1:
            actual_index = index // self.num_samples_per_image
        else:
            actual_index = index

        item = self.data[actual_index]

        # Load all fields
        loaded_data = {}
        try:
            for field in self.fields:
                loaded_data[field] = self.image_loader(item[field])
            loaded_data["name"] = item["name"]
        except Exception as e:
            raise RuntimeError(f"Failed to load data for {item['name']}: {e}")

        # Apply default transforms
        data = self.default_transforms(loaded_data)

        # Apply augmentation if training
        if self.augmentation:
            data = self.augmentation_transforms(data)

            # RandCropByPosNegLabeld returns list of dicts when num_samples > 1
            # Extract the appropriate sample
            if self.num_samples_per_image > 1 and isinstance(data, list):
                sample_idx = index % self.num_samples_per_image
                data = data[sample_idx]

        # 기본 geometry가 없으면 hard label을 geometry로 제공 (flow/확장 모델 호환용)
        if 'geometry' not in data and 'label' in data:
            data['geometry'] = data['label'].float()

        # Always attach a coordinate grid for consumers that need it (e.g., flow models)
        if 'coordinate' not in data and 'image' in data:
            c, h, w = data['image'].shape
            device = data['image'].device if hasattr(data['image'], 'device') else None
            # Normalized [-1, 1] meshgrid, shape (2, H, W)
            y = torch.linspace(-1, 1, steps=h, device=device)
            x = torch.linspace(-1, 1, steps=w, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            data['coordinate'] = torch.stack([xx, yy], dim=0)

        return data


class BaseOCTDataModule(L.LightningDataModule, ABC):
    """
    Base DataModule for OCT datasets.
    
    PyTorch Lightning DataModule benefits:
    1. Separates data loading logic from model code
    2. Handles data splitting for distributed training automatically
    3. Systematically manages train/val/test dataloaders
    4. Perfect integration with Lightning Trainer
    5. Improves code consistency and readability
    
    Subclasses must implement:
    - create_train_dataset(): Create training dataset (may combine multiple sources)
    
    Subclasses should set:
    - dataset_class: The Dataset class to use (e.g., OCTADataset, ROSSADataset)
    """

    dataset_class = None  # Must be set by subclass

    @abstractmethod
    def create_train_dataset(self):
        """
        Create training dataset.
        
        This allows subclasses to implement special logic like:
        - Single directory: return self.dataset_class(self.train_dir, ...)
        - Multiple directories: return ConcatDataset([dataset1, dataset2])
        
        Returns:
            Dataset or ConcatDataset: Training dataset
        """
        pass

    def __init__(self, train_dir, val_dir, test_dir, crop_size, train_bs=8,
                 num_samples_per_image=1, name="base"):
        """
        Args:
            train_dir: Training data path (may be None for special cases)
            val_dir: Validation data path
            test_dir: Test data path
            crop_size: Crop size for training
            train_bs: Training batch size
            num_samples_per_image: Samples per image for augmentation (default: 1)
            name: Dataset name
        """
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.crop_size = crop_size
        self.train_bs = train_bs
        self.num_samples_per_image = num_samples_per_image
        self.name = name

        self.save_hyperparameters()

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Setup datasets for training/validation/testing.
        
        Called automatically by Lightning Trainer.
        
        Args:
            stage: 'fit', 'test', 'predict', etc. (usually None)
        """
        # Create training dataset (subclass-specific logic)
        self.train_dataset = self.create_train_dataset()

        # Create validation and test datasets (subclass may override)
        self.val_dataset = self.create_val_dataset()
        self.test_dataset = self.create_test_dataset()

    def create_val_dataset(self):
        """Create validation dataset (can be overridden by subclasses)."""
        return self.dataset_class(
            self.val_dir,
            augmentation=False,
            crop_size=self.crop_size,
            num_samples_per_image=1  # val/test always use 1 sample
        )

    def create_test_dataset(self):
        """Create test dataset (can be overridden by subclasses)."""
        return self.dataset_class(
            self.test_dir,
            augmentation=False,
            crop_size=self.crop_size,
            num_samples_per_image=1  # val/test always use 1 sample
        )

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool = False, sampler=None):
        """
        Create DataLoader with common settings.
        
        DataLoader parameters:
        - batch_size: Number of samples per batch
        - shuffle: Whether to shuffle data order (True for training only)
        - num_workers: Number of worker processes for data loading (parallel processing)
        - pin_memory: Speed up GPU transfer (recommended True for CUDA)
        - prefetch_factor: Number of batches to prefetch (memory vs speed tradeoff)
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=16,  # Optimized for single GPU (too many workers can cause overhead)
            pin_memory=True,  # Enable for faster GPU transfer
            prefetch_factor=4,  # Prefetch more batches for better pipeline
            persistent_workers=True,  # Keep workers alive between epochs
        )

    def train_dataloader(self):
        """Return training DataLoader (shuffled)"""
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return self._create_dataloader(self.train_dataset, self.train_bs, shuffle=True, sampler=sampler)

    def val_dataloader(self):
        """Return validation DataLoader (not shuffled)
        
        Note: DDP 사용 시 validation 데이터가 GPU별로 분할되면,
        rank 0가 처리하지 않은 이미지는 로깅되지 않습니다.
        따라서 distributed_sampler를 사용하지 않도록 설정합니다.
        (Lightning의 replace_sampler_ddp=False 옵션 사용)
        """
        return self._create_dataloader(self.val_dataset, 1, shuffle=False)

    def test_dataloader(self):
        """Return test DataLoader (not shuffled)"""
        return self._create_dataloader(self.test_dataset, 1, shuffle=False)
