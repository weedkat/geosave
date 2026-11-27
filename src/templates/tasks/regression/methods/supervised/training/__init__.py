"""Supervised pixelwise regression: physical values per pixel, such as biomass."""

from .datamodule import DataModule, Sample, SampleDataset
from .module import Module

__all__ = ["DataModule", "Module", "Sample", "SampleDataset"]
