from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from generated_kits.pretraining.mim import training as mim


FLIP = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]


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
    data = mim.DataModule(
        train=Chips(),
        val=Chips(),
        **options,
    )
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def _unlabeled(times: int = 0) -> mim.Sample:
    shape = (2, times, 2, 8, 8) if times else (2, 2, 8, 8)
    image = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    return {"inputs": {"image": image}}


def test_training_batches_share_geometry_across_all_instants(tmp_path) -> None:
    data = _data(tmp_path, True, augmentations=FLIP)
    image = _unlabeled(times=3)["inputs"]["image"]

    batch = data.on_after_batch_transfer(_unlabeled(times=3), 0)

    torch.testing.assert_close(batch["inputs"]["image"], image.flip(-1))
    assert set(batch) == {"inputs"}
