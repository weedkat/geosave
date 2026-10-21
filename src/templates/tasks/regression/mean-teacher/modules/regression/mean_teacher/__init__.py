"""Mean Teacher regression: an EMA teacher's weak-view values supervise strong views."""

from .data import DataModule, TrainingBatch, Unlabeled, UnlabeledDataset, Views
from .module import Module

__all__ = [
    "DataModule",
    "Module",
    "TrainingBatch",
    "Unlabeled",
    "UnlabeledDataset",
    "Views",
]
