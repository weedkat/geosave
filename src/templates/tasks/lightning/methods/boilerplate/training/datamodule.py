"""In-memory data for the ordinary Lightning task template."""

from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from .datasets import ExampleDataset


class DataModule(LightningDataModule):
    """Serve deterministic examples through ordinary PyTorch data loaders.

    Replace the dataset construction in ``setup`` to connect the workspace to
    application data; the loaders deliberately remain standard PyTorch loaders.
    """

    def __init__(
        self,
        train_size: int = 16,
        val_size: int = 8,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data: ExampleDataset | None = None
        self.val_data: ExampleDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create the deterministic training and validation datasets."""
        if stage in (None, "fit"):
            self.train_data = ExampleDataset(self.train_size)
            self.val_data = ExampleDataset(self.val_size)

    def train_dataloader(self) -> DataLoader[tuple[dict[str, Tensor], Tensor]]:
        """Return the training data loader."""
        if self.train_data is None:
            self.setup("fit")
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[tuple[dict[str, Tensor], Tensor]]:
        """Return the validation data loader."""
        if self.val_data is None:
            self.setup("fit")
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
