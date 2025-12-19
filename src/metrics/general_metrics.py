"""General segmentation metrics using torchmetrics."""

from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)
from torchmetrics.segmentation.dice import DiceScore as Dice

# Re-export with simpler names
Precision = MulticlassPrecision
Recall = MulticlassRecall
Specificity = MulticlassSpecificity
JaccardIndex = MulticlassJaccardIndex

__all__ = [
    'Dice',
    'Precision',
    'Recall',
    'Specificity',
    'JaccardIndex',
]
