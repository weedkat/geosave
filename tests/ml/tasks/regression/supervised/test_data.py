from __future__ import annotations

from types import SimpleNamespace

import torch

from geosave.ml.tasks.regression import supervised
from tests.ml.tasks.segmentation.supervised.test_data import FLIP, Chips, _batch, _spec


def _training(tmp_path, **options: object) -> supervised.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = supervised.DataModule(spec=path, train=Chips(), val=Chips(), **options)
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    return data


def _values() -> torch.Tensor:
    values = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    values[0, 0, 0, 0] = torch.nan
    return values


def test_target_values_move_with_the_pixels(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP)
    image = _batch()["inputs"]["image"]

    batch = data.on_after_batch_transfer({**_batch(), "target": _values()}, 0)

    torch.testing.assert_close(batch["inputs"]["image"], image.flip(-1))
    torch.testing.assert_close(batch["target"], _values().flip(-1), equal_nan=True)


def test_values_an_augmentation_invents_are_unknown_not_zero(tmp_path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(tmp_path, augmentations=shift)

    batch = data.on_after_batch_transfer(
        {**_batch(), "target": torch.ones(2, 1, 8, 8)}, 0
    )

    target = batch["target"]
    assert target.isnan().any()
    assert set(target[target.isfinite()].unique().tolist()) == {1.0}
