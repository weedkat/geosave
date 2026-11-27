"""Read this project's samples; implement the Dataset for your data source."""

from torch.utils.data import Dataset

from .fixmatch_datamodule import Unlabeled


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
