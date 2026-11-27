from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader

from geosave.ml.torch.training.callbacks.segmentation import DensePredictionLogger


def test_validation_outputs_are_logged_as_a_prediction_panel(tmp_path: Path) -> None:
    class Evaluation(LightningModule):
        def validation_step(self, batch, batch_idx):
            image, target = batch
            return {"logits": image, "label": target}

    loader = DataLoader([(torch.ones(2, 4, 4), torch.zeros(4, 4, dtype=torch.long))])
    logger = TensorBoardLogger(str(tmp_path))
    trainer = Trainer(
        accelerator="cpu",
        logger=logger,
        callbacks=[DensePredictionLogger({0: "#000000"})],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.validate(Evaluation(), dataloaders=loader)
    logger.experiment.flush()
    events = EventAccumulator(logger.log_dir).Reload()
    (panel,) = events.Images("val/dataloader_0/prediction")
    assert panel.step == 0
    assert panel.width > 0 and panel.height > 0


@pytest.mark.parametrize("reason", ["sanity", "rank", "logger", "batch", "epoch"])
def test_skipped_batches_do_not_render_or_require_outputs(reason: str) -> None:
    callback = DensePredictionLogger({0: "#000000"})
    trainer = Mock(spec=Trainer)
    trainer.sanity_checking = reason == "sanity"
    trainer.is_global_zero = reason != "rank"
    trainer.current_epoch = 1 if reason == "epoch" else 0
    trainer.loggers = [] if reason == "logger" else [Mock(spec=TensorBoardLogger)]

    callback.on_validation_batch_end(
        trainer, LightningModule(), None, None, 1 if reason == "batch" else 0
    )
    for logger in trainer.loggers:
        logger.experiment.add_image.assert_not_called()


def test_loader_images_have_distinct_names_and_argmax_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = DensePredictionLogger({0: "#000000", 1: "#ffffff"})
    trainer = Mock(spec=Trainer)
    trainer.sanity_checking = False
    trainer.is_global_zero = True
    trainer.current_epoch = 2
    logger = Mock(spec=TensorBoardLogger)
    trainer.loggers = [logger]
    label = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.tensor([[[[3.0, 0.0], [1.0, 2.0]], [[0.0, 4.0], [2.0, 0.0]]]])
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def render(actual_label: torch.Tensor, actual_preds: torch.Tensor) -> np.ndarray:
        torch.testing.assert_close(actual_label, label[0])
        torch.testing.assert_close(actual_preds, label[0])
        return image

    monkeypatch.setattr(callback, "_render", render)
    for loader in (0, 1):
        callback.on_validation_batch_end(
            trainer,
            LightningModule(),
            {"logits": logits, "label": label},
            None,
            0,
            loader,
        )
    assert [call.args[0] for call in logger.experiment.add_image.call_args_list] == [
        "val/dataloader_0/prediction",
        "val/dataloader_1/prediction",
    ]
    assert logger.experiment.add_image.call_args.args[1] is image
    assert logger.experiment.add_image.call_args.args[2] == 2
