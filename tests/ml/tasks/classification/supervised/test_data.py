from __future__ import annotations

from types import SimpleNamespace

import torch

from geosave_engine.ml.tasks.classification import supervised
from tests.ml.tasks.segmentation.supervised.test_data import FLIP, Chips, _batch, _spec


def test_training_batches_move_pixels_and_validity_together(tmp_path) -> None:
    path = _spec().save(tmp_path / "model_spec.yaml")
    data = supervised.DataModule(
        spec=path, train=Chips(), val=Chips(), augmentations=FLIP
    )
    data.trainer = SimpleNamespace(training=True)  # type: ignore[assignment]
    batch = _batch()
    batch["valid"][:, :, 0] = False
    image = batch["inputs"]["image"].clone()
    target = torch.tensor([1, 2])

    finished = data.on_after_batch_transfer({**batch, "target": target}, 0)

    torch.testing.assert_close(finished["inputs"]["image"], image.flip(-1))
    assert torch.equal(finished["target"], target)
    moved = finished["valid"]
    assert not moved[:, :, -1].any() and moved[:, :, :-1].all()
