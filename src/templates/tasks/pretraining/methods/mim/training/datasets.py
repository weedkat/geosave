"""Read this project's samples; implement the Dataset for your data source."""

from torch.utils.data import Dataset

from .datamodule import Sample


class MyDataset(Dataset[Sample]):
    """Read one sample per index from a named split.

    Select the split, call processing.preprocessing.preprocess on the source,
    and return named tensors in the layout declared by Sample.
    """

    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        raise NotImplementedError("Implement the Dataset for your data source.")

    def __getitem__(self, index: int) -> Sample:
        """Read one sample for this training method."""
        raise NotImplementedError("Implement the Dataset for your data source.")
