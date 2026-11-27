"""Postprocessors, one per task: they turn chip outputs into a product."""

from .base import PostprocessOutput, Postprocessor
from .regression import RegressionPostprocessor
from .segmentation import SegmentationPostprocessor

__all__ = ["PostprocessOutput", "Postprocessor", "RegressionPostprocessor", "SegmentationPostprocessor"]
