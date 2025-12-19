"""Dataset registry with metadata support.

Usage:
    # 데코레이터로 등록 (권장)
    @register_dataset(
        name='my_dataset',
        modality='OCTA',
        resolution=(512, 512),
        num_train=100,
        num_val=20,
        num_test=30,
        description='My custom dataset',
    )
    class MyDataModule(BaseDataModule):
        pass
    
    # 조회
    dataset_cls = DATASET_REGISTRY.get('my_dataset')
    dataset_info = get_dataset_info('my_dataset')
"""

from dataclasses import dataclass
from typing import Type, Optional, Tuple, List

from .base import Registry, DATASET_REGISTRY


@dataclass
class DatasetInfo:
    """Dataset metadata for automatic configuration."""
    name: str
    class_ref: Type  # DataModule class
    modality: str  # 'OCTA', 'OCT', 'XCA', 'Fundus', etc.
    resolution: Tuple[int, int]  # (height, width)
    num_train: int
    num_val: int
    num_test: int
    description: str
    default_crop_size: int = 224
    default_batch_size: int = 8
    default_train_dir: Optional[str] = None
    default_val_dir: Optional[str] = None
    default_test_dir: Optional[str] = None


# =============================================================================
# Built-in dataset metadata (for datasets registered without metadata)
# =============================================================================

BUILTIN_DATASET_METADATA = {
    'octa500_3m': {
        'modality': 'OCTA',
        'resolution': (304, 304),
        'num_train': 200,
        'num_val': 50,
        'num_test': 50,
        'description': 'OCTA-500 3x3mm vessel segmentation',
        'default_crop_size': 224,
        'default_batch_size': 8,
        'default_train_dir': 'data/OCTA500_3M/train',
        'default_val_dir': 'data/OCTA500_3M/val',
        'default_test_dir': 'data/OCTA500_3M/test',
    },
    'octa500_6m': {
        'modality': 'OCTA',
        'resolution': (400, 400),
        'num_train': 200,
        'num_val': 50,
        'num_test': 50,
        'description': 'OCTA-500 6x6mm vessel segmentation',
        'default_crop_size': 224,
        'default_batch_size': 8,
        'default_train_dir': 'data/OCTA500_6M/train',
        'default_val_dir': 'data/OCTA500_6M/val',
        'default_test_dir': 'data/OCTA500_6M/test',
    },
    'rossa': {
        'modality': 'OCTA',
        'resolution': (304, 304),
        'num_train': 200,
        'num_val': 50,
        'num_test': 50,
        'description': 'ROSSA FAZ segmentation',
        'default_crop_size': 224,
        'default_batch_size': 8,
        'default_train_dir': 'data/ROSSA/train_manual',
        'default_val_dir': 'data/ROSSA/val',
        'default_test_dir': 'data/ROSSA/test',
    },
    'xca': {
        'modality': 'XCA',
        'resolution': (512, 512),
        'num_train': 155,
        'num_val': 20,
        'num_test': 46,
        'description': 'XCA full vessel segmentation dataset',
        'default_crop_size': 320,
        'default_batch_size': 4,
        'default_train_dir': 'data/xca_full/train',
        'default_val_dir': 'data/xca_full/val',
        'default_test_dir': 'data/xca_full/test',
    },
}


def register_dataset(
    name: str,
    modality: str,
    resolution: Tuple[int, int],
    num_train: int,
    num_val: int,
    num_test: int,
    description: str,
    default_crop_size: int = 224,
    default_batch_size: int = 8,
    default_train_dir: Optional[str] = None,
    default_val_dir: Optional[str] = None,
    default_test_dir: Optional[str] = None,
):
    """Decorator to register a dataset with metadata.
    
    Example:
        @register_dataset(
        name='xca',
        modality='XCA',
        resolution=(512, 512),
        num_train=155,
        num_val=20,
        num_test=46,
            description='XCA vessel segmentation',
        default_crop_size=320,
        default_train_dir='data/xca_full/train',
        )
        class XCA_DataModule(BaseDataModule):
            pass
    """
    def decorator(cls: Type) -> Type:
        metadata = {
            'modality': modality,
            'resolution': resolution,
            'num_train': num_train,
            'num_val': num_val,
            'num_test': num_test,
            'description': description,
            'default_crop_size': default_crop_size,
            'default_batch_size': default_batch_size,
            'default_train_dir': default_train_dir,
            'default_val_dir': default_val_dir,
            'default_test_dir': default_test_dir,
        }
        DATASET_REGISTRY.register(name=name, obj=cls, metadata=metadata)
        return cls
    return decorator


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Get dataset metadata as DatasetInfo dataclass.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        DatasetInfo with all metadata
    
    Raises:
        KeyError: If dataset not found
    """
    cls = DATASET_REGISTRY.get(dataset_name)
    
    # Try to get metadata from registry first
    try:
        metadata = DATASET_REGISTRY.get_metadata(dataset_name)
        if metadata:
            metadata = dict(metadata)  # Copy to avoid modifying original
        else:
            metadata = {}
    except KeyError:
        metadata = {}
    
    # If metadata is empty, use built-in metadata
    if not metadata and dataset_name in BUILTIN_DATASET_METADATA:
        metadata = dict(BUILTIN_DATASET_METADATA[dataset_name])
    
    # Ensure all required fields have defaults
    metadata.setdefault('modality', 'unknown')
    metadata.setdefault('resolution', (0, 0))
    metadata.setdefault('num_train', 0)
    metadata.setdefault('num_val', 0)
    metadata.setdefault('num_test', 0)
    metadata.setdefault('description', '')
    metadata.setdefault('default_crop_size', 224)
    metadata.setdefault('default_batch_size', 8)
    metadata.setdefault('default_train_dir', None)
    metadata.setdefault('default_val_dir', None)
    metadata.setdefault('default_test_dir', None)
    
    return DatasetInfo(
        name=dataset_name,
        class_ref=cls,
        **metadata
    )


def list_datasets(modality: Optional[str] = None) -> List[str]:
    """List all available datasets, optionally filtered by modality.
    
    Args:
        modality: Filter by modality (e.g., 'OCTA', 'XCA')
    
    Returns:
        List of dataset names
    """
    all_datasets = DATASET_REGISTRY.list()
    
    if modality is None:
        return all_datasets
    
    # Filter by modality using built-in metadata
    result = []
    for name in all_datasets:
        # Check registry metadata first
        try:
            meta = DATASET_REGISTRY.get_metadata(name)
            if meta and meta.get('modality') == modality:
                result.append(name)
                continue
        except KeyError:
            pass
        
        # Fall back to built-in metadata
        if name in BUILTIN_DATASET_METADATA:
            if BUILTIN_DATASET_METADATA[name].get('modality') == modality:
                result.append(name)
    
    return result
