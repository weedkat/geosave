"""Supervised segmentation: the training module and the data that feeds it."""

from .data import DataModule, Sample
from .module import Module

__all__ = ["DataModule", "Module", "Sample"]
