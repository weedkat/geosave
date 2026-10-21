"""Deterministic examples for the editable Lightning starter."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ExampleDataset(Dataset[tuple[dict[str, Tensor], Tensor]]):
    """Return deterministic binary-classification examples from an index.

    Replace this dataset with the application's data source while preserving the
    ``(inputs, target)`` batch shape consumed by ``CustomTask``.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        """Return the number of examples."""
        return self.size

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Tensor]:
        """Return fixed features and a target derived from ``index``."""
        generator = torch.Generator().manual_seed(index)
        inputs = torch.rand(4, generator=generator)
        target = (inputs.sum() > 2).long()
        return {"features": inputs}, target
