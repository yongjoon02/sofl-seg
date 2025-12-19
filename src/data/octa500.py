"""OCTA500 dataset with base class inheritance."""
from src.data.base_dataset import BaseOCTDataModule, BaseOCTDataset
from src.registry import DATASET_REGISTRY
from src.utils.visualize_dataloader import visualize_dataset


class OCTADataset(BaseOCTDataset):
    """OCTA500 Dataset with 2 fields: image, label (soft labels generated during training)"""

    def get_data_fields(self) -> list[str]:
        """Only load image and label - soft labels will be generated dynamically"""
        return ['image', 'label']


class OCTADataModule(BaseOCTDataModule):
    """OCTA500 DataModule using single training directory"""

    dataset_class = OCTADataset

    def create_train_dataset(self):
        """Create training dataset from single directory"""
        return self.dataset_class(
            self.train_dir,
            augmentation=True,
            crop_size=self.crop_size,
            num_samples_per_image=self.num_samples_per_image
        )


@DATASET_REGISTRY.register(name='octa500_3m')
class OCTA500_3M_DataModule(OCTADataModule):
    """OCTA500 3mm DataModule"""
    def __init__(self, train_dir="data/OCTA500_3M/train", val_dir="data/OCTA500_3M/val",
                 test_dir="data/OCTA500_3M/test", crop_size=128, train_bs=8, num_samples_per_image=1):
        super().__init__(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
                        crop_size=crop_size, train_bs=train_bs,
                        num_samples_per_image=num_samples_per_image, name='octa500_3m')


@DATASET_REGISTRY.register(name='octa500_6m')
class OCTA500_6M_DataModule(OCTADataModule):
    """OCTA500 6mm DataModule"""
    def __init__(self, train_dir="data/OCTA500_6M/train", val_dir="data/OCTA500_6M/val",
                 test_dir="data/OCTA500_6M/test", crop_size=128, train_bs=8, num_samples_per_image=1):
        super().__init__(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir,
                        crop_size=crop_size, train_bs=train_bs,
                        num_samples_per_image=num_samples_per_image, name='octa500_6m')


if __name__ == "__main__":
    datamodule = DATASET_REGISTRY.get('octa500_3m')()
    datamodule.setup()
    visualize_dataset(datamodule.train_dataloader(), "octa500_3m_train")
    visualize_dataset(datamodule.val_dataloader(), "octa500_3m_val")
    visualize_dataset(datamodule.test_dataloader(), "octa500_3m_test")

    datamodule = DATASET_REGISTRY.get('octa500_6m')()
    datamodule.setup()
    visualize_dataset(datamodule.train_dataloader(), "octa500_6m_train")
    visualize_dataset(datamodule.val_dataloader(), "octa500_6m_val")
    visualize_dataset(datamodule.test_dataloader(), "octa500_6m_test")
