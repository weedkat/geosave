"""Supervised segmentation: the training module and the data that feeds it."""

from .datamodule import DataModule, Sample, SampleDataset
from .module import Module

__all__ = ["DataModule", "Module", "Sample", "SampleDataset"]
