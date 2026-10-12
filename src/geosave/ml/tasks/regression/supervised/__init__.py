"""Supervised pixelwise regression: physical values per pixel, such as biomass."""

from .data import DataModule, Sample
from .module import Module

__all__ = ["DataModule", "Module", "Sample"]
