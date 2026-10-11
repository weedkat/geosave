from __future__ import annotations

from types import SimpleNamespace

import torch

from geosave_engine.ml.regression import supervised
from geosave_engine.model.spec import Ref
from tests.ml.segmentation.supervised.test_data import FLIP, _batch, _manifest, _spec


def _training(tmp_path, **options: object) -> supervised.DataModule:
    """Build only the batch pipelines of a module that is fitting."""
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = supervised.DataModule(spec=path, train=tmp_path, val=tmp_path, **options)
    data.setup("predict")
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    return data


def _values() -> torch.Tensor:
    values = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    values[0, 0, 0, 0] = torch.nan
    return values


def test_target_values_move_with_the_pixels(tmp_path) -> None:
    data = _training(tmp_path, augmentations=FLIP)
    model_inputs, _, ids, valid = _batch()

    finished, target, _, _ = data.on_after_batch_transfer(
        (model_inputs, _values(), ids, valid), 0
    )

    torch.testing.assert_close(finished["image"], model_inputs["image"].flip(-1))
    torch.testing.assert_close(target, _values().flip(-1), equal_nan=True)


def test_values_an_augmentation_invents_are_unknown_not_zero(tmp_path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(tmp_path, augmentations=shift)
    model_inputs, _, ids, valid = _batch()
    ones = torch.ones(2, 1, 8, 8)

    _, target, _, _ = data.on_after_batch_transfer((model_inputs, ones, ids, valid), 0)

    assert target.isnan().any()
    assert set(target[target.isfinite()].unique().tolist()) == {1.0}


def test_a_dataset_needs_its_target_collection(tmp_path) -> None:
    spec = _spec(preprocessing={}, context=None, inputs={"image": Ref("optical")})

    dataset = supervised.Dataset(_manifest(tmp_path), spec, target="label")

    assert len(dataset) == 32
