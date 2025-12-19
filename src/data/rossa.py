"""ROSSA Retinal Vessel Segmentation Dataset

ROSSA dataset with manual and SAM annotations.
No label_prob - simpler than OCTA dataset.
"""
from torch.utils.data import ConcatDataset

from src.data.base_dataset import BaseOCTDataModule, BaseOCTDataset
from src.registry import DATASET_REGISTRY
from src.utils.visualize_dataloader import visualize_dataset


class ROSSADataset(BaseOCTDataset):
    """ROSSA Dataset with 2 fields: image, label (soft labels generated during training)"""

    def get_data_fields(self) -> list[str]:
        """Only load image and label - soft labels will be generated dynamically"""
        return ['image', 'label']


class ROSSADataModule(BaseOCTDataModule):
    """ROSSA DataModule with optional SAM data"""

    dataset_class = ROSSADataset

    def __init__(self,
                 train_manual_dir="data/ROSSA/train_manual",
                 train_sam_dir="data/ROSSA/train_sam",
                 val_dir="data/ROSSA/val",
                 test_dir="data/ROSSA/test",
                 crop_size=128,
                 train_bs=8,
                 num_samples_per_image=1,
                 use_sam: bool = False):
        """
        Args:
            train_manual_dir: Manual annotation training data
            train_sam_dir: SAM annotation training data
            val_dir: Validation data
            test_dir: Test data
            crop_size: Crop size for training
            train_bs: Training batch size
            num_samples_per_image: Samples per image for augmentation
            use_sam: Whether to include SAM-annotated data (default: False, manual only)
        """
        # Store ROSSA-specific directories
        self.train_manual_dir = train_manual_dir
        self.train_sam_dir = train_sam_dir
        self.use_sam = use_sam

        # Call parent init with None for train_dir (we use manual+sam instead)
        super().__init__(
            train_dir=None,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            name='rossa'
        )

    def create_train_dataset(self):
        """Create training dataset (manual only by default, optionally with SAM)"""
        # Train: Manual annotations (always included)
        train_manual_dataset = self.dataset_class(
            self.train_manual_dir,
            augmentation=True,
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )

        print("ROSSA Dataset loaded:")
        print(f"  Train (manual): {len(train_manual_dataset)} samples")

        if self.use_sam:
            # Optionally add SAM annotations
            train_sam_dataset = self.dataset_class(
                self.train_sam_dir,
                augmentation=True,
                crop_size=self.crop_size,
                num_samples_per_image=self.num_samples_per_image
            )
            # Concatenate manual and SAM datasets
            combined_dataset = ConcatDataset([train_manual_dataset, train_sam_dataset])
            print(f"  Train (SAM): {len(train_sam_dataset)} samples")
            print(f"  Train (total): {len(combined_dataset)} samples")
            return combined_dataset
        else:
            print("  (SAM data disabled, use_sam=False)")
            return train_manual_dataset


@DATASET_REGISTRY.register(name='rossa')
class ROSSA_DataModule(ROSSADataModule):
    """ROSSA DataModule registered in dataset registry"""
    def __init__(self,
                 train_manual_dir="data/ROSSA/train_manual",
                 train_sam_dir="data/ROSSA/train_sam",
                 val_dir="data/ROSSA/val",
                 test_dir="data/ROSSA/test",
                 crop_size=128,
                 train_bs=8,
                 num_samples_per_image=1,
                 use_sam: bool = False):
        super().__init__(
            train_manual_dir=train_manual_dir,
            train_sam_dir=train_sam_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            crop_size=crop_size,
            train_bs=train_bs,
            num_samples_per_image=num_samples_per_image,
            use_sam=use_sam
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ROSSA Dataset")
    print("=" * 70)

    # Test 1: Manual only (default)
    print("\n=== Test 1: use_sam=False (default) ===")
    dm = ROSSA_DataModule(use_sam=False)
    dm.setup()

    # Test 2: With SAM data
    print("\n=== Test 2: use_sam=True ===")
    dm2 = ROSSA_DataModule(use_sam=True)
    dm2.setup()

    # Visualize default (manual only)
    visualize_dataset(dm.train_dataloader(), "rossa_train")
    visualize_dataset(dm.val_dataloader(), "rossa_val")
    visualize_dataset(dm.test_dataloader(), "rossa_test")

    print("\n✓ ROSSA dataset works correctly!")
