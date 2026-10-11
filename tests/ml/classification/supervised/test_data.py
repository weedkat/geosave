from __future__ import annotations

from types import SimpleNamespace

import torch

from geosave_engine.ml.classification import supervised
from geosave_engine.model.spec import Ref
from tests.ml.segmentation.supervised.test_data import FLIP, _batch, _manifest, _spec


def _plain_spec():
    """A spec reading the optical bands as they are."""
    return _spec(preprocessing={}, context=None, inputs={"image": Ref("optical")})


def test_each_chip_takes_the_class_of_its_window(tmp_path) -> None:
    manifest = _manifest(tmp_path)
    manifest["class_id"] = manifest["id"].str.split("/").str[0].map({"s0": 3, "s1": 5})

    dataset = supervised.Dataset(manifest, _plain_spec())

    assert dataset.labels.dtype == torch.int64
    assert set(dataset.labels.tolist()) == {3, 5}
    assert len(dataset.labels) == len(dataset)


def test_training_batches_move_pixels_and_validity_together(tmp_path) -> None:
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = supervised.DataModule(
        spec=path, train=tmp_path, val=tmp_path, augmentations=FLIP
    )
    data.setup("predict")
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    model_inputs, _, ids, valid = _batch()
    valid[:, :, 0] = False
    target = torch.tensor([1, 2])

    finished, labels, _, moved = data.on_after_batch_transfer(
        (model_inputs, target, ids, valid), 0
    )

    torch.testing.assert_close(finished["image"], model_inputs["image"].flip(-1))
    assert torch.equal(labels, target)
    assert not moved[:, :, -1].any() and moved[:, :, :-1].all()
