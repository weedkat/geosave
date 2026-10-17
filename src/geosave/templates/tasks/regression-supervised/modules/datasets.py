"""The workspace's Dataset: read one chip per index, return a `Sample`."""

from geosave.ml.tasks.regression.supervised import Sample


class MyDataset:
    """Replace with a reader of this workspace's data; `split` is train or val."""

    def __init__(self, split: str) -> None:
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError
