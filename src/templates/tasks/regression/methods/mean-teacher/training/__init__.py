"""Mean Teacher regression: an EMA teacher's weak-view values supervise strong views."""

from .teacher_datamodule import (
    DataModule,
    TrainingBatch,
    Unlabeled,
    UnlabeledDataset,
    Views,
)
from .teacher_module import Module

__all__ = [
    "DataModule",
    "Module",
    "TrainingBatch",
    "Unlabeled",
    "UnlabeledDataset",
    "Views",
]
