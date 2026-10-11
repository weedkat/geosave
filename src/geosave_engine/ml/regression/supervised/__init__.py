"""Supervised pixelwise regression: physical values per pixel, such as biomass."""

from .data import DataModule, Dataset
from .module import Module

__all__ = ["DataModule", "Dataset", "Module"]
