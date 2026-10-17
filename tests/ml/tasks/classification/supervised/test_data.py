from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from geosave.ml.tasks.classification import supervised
from tests.ml.tasks.segmentation.supervised.test_data import FLIP, _batch, _spec


class Chips(Dataset[supervised.Sample]):
    """Source yielding this method's Sample contract."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> supervised.Sample:
        return {
            "inputs": {"image": torch.full((2, 8, 8), float(index))},
            "target": torch.tensor(index % 2),
        }


def test_training_batches_move_pixels_and_preserve_class_targets(tmp_path) -> None:
    data = supervised.DataModule(_spec(), Chips(), Chips(), augmentations=FLIP)
    data.trainer = SimpleNamespace(training=True)
    batch = _batch()
    image = batch["inputs"]["image"].clone()
    target = torch.tensor([1, 2])

    result = data.on_after_batch_transfer({**batch, "target": target}, 0)

    torch.testing.assert_close(result["inputs"]["image"], image.flip(-1))
    assert torch.equal(result["target"], target)
    assert set(result) == {"inputs", "target"}
