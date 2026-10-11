from .classification import ClassificationHead
from .dense import DenseHead
from .regression import RegressionHead
from .segmentation import SegmentationHead

__all__ = [
    "ClassificationHead",
    "DenseHead",
    "RegressionHead",
    "SegmentationHead",
]
