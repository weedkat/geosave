from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from modules.pretraining import mim
from tests.templates.tasks.segmentation.supervised.test_data import (
    FLIP,
    NORMALIZE,
    _batch,
    _spec,
)

pytestmark = pytest.mark.skip(
    reason="Template processing awaits the ModelSpec redesign."
)


class Chips(Dataset[mim.Sample]):
    """Source yielding this method's Sample contract."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> mim.Sample:
        return {
            "inputs": {"image": torch.full((2, 8, 8), float(index))},
        }


def _data(tmp_path, training: bool, **options: object) -> mim.DataModule:
    """Build only the batch pipelines."""
    path = _spec(transforms=NORMALIZE).save(tmp_path / "model_spec.yaml")
    data = mim.DataModule(spec=path, train=Chips(), val=Chips(), **options)
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def _unlabeled(times: int = 0) -> mim.Sample:
    batch = _batch(times)
    return {"inputs": batch["inputs"]}


def test_training_batches_are_augmented_then_transformed(tmp_path) -> None:
    data = _data(tmp_path, True, augmentations=FLIP)
    image = _unlabeled(times=3)["inputs"]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    batch = data.on_after_batch_transfer(_unlabeled(times=3), 0)

    torch.testing.assert_close(batch["inputs"]["image"], (image.flip(-1) - mean) / std)
    assert set(batch) == {"inputs"}
