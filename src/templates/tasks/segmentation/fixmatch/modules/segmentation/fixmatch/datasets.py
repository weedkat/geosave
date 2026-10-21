"""Read this project's samples; implement the Dataset for your data source."""

from torch.utils.data import Dataset

from .supervised.data import Sample
from .data import Unlabeled


class MyDataset(Dataset[Sample]):
    """Read one sample per index from a named split.

    Implement file discovery, raster preparation, and sample reads here.
    """

    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        raise NotImplementedError("Implement the Dataset for your data source.")

    def __getitem__(self, index: int) -> Sample:
        """Read one sample for this training method."""
        raise NotImplementedError("Implement the Dataset for your data source.")


class MyUnlabeledDataset(Dataset[Unlabeled]):
    """Read unlabeled inputs for the semi-supervised method."""

    def __len__(self) -> int:
        """Return the number of unlabeled samples."""
        raise NotImplementedError(
            "Implement the unlabeled Dataset for your data source."
        )

    def __getitem__(self, index: int) -> Unlabeled:
        """Read named inputs without a target."""
        raise NotImplementedError(
            "Implement the unlabeled Dataset for your data source."
        )
