"""Exercise each task's Dataset-to-Trainer contract."""

import pytest
import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

from geosave.ml.tasks.classification import supervised as classification
from geosave.ml.tasks.pretraining import mim
from geosave.ml.tasks.regression import mean_teacher, supervised as regression
from geosave.ml.tasks.segmentation import fixmatch, supervised as segmentation
from tests.ml.tasks.classification.supervised.test_data import Chips as Classes
from tests.ml.tasks.classification.supervised.test_module import STAGES as CLASSIFIER
from tests.ml.tasks.pretraining.mim.test_data import Chips as Scenes
from tests.ml.tasks.pretraining.mim.test_module import STAGES as RECONSTRUCTOR
from tests.ml.tasks.regression.mean_teacher.test_data import Scenes as UnlabeledValues
from tests.ml.tasks.regression.supervised.test_data import Chips as Values
from tests.ml.tasks.regression.supervised.test_module import STAGES as REGRESSOR
from tests.ml.tasks.segmentation.fixmatch.test_data import Scenes as UnlabeledLabels
from tests.ml.tasks.segmentation.fixmatch.test_module import STAGES as SEGMENTER
from tests.ml.tasks.segmentation.supervised.test_data import (
    FLIP,
    Chips as Labels,
    _spec,
)


@pytest.mark.parametrize(
    "task,source,stages,unlabeled",
    [
        (classification, Classes, CLASSIFIER, None),
        (segmentation, Labels, SEGMENTER, None),
        (regression, Values, REGRESSOR, None),
        (mim, Scenes, RECONSTRUCTOR, None),
        (fixmatch, Labels, SEGMENTER, UnlabeledLabels),
        (mean_teacher, Values, REGRESSOR, UnlabeledValues),
    ],
)
def test_trainer_consumes_model_ready_samples(
    task, source, stages, unlabeled, tmp_path
):
    options = {"unlabeled": unlabeled()} if unlabeled else {}
    data = task.DataModule(
        _spec(),
        source(),
        source(),
        augmentations=FLIP,
        batch_size=2,
        num_workers=0,
        **options,
    )
    module = task.Module(model_chain=stages)
    trainer = Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(module, datamodule=data)

    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["train_loss"])
    assert set(source()[0]) == ({"inputs"} if task is mim else {"inputs", "target"})


class SparseLabels(Dataset[segmentation.Sample]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> segmentation.Sample:
        return {
            "inputs": {"image": torch.ones(2, 2, 2)},
            "target": torch.full((2, 2), 255 if index == 0 else 0, dtype=torch.long),
        }


class SparseValues(Dataset[regression.Sample]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> regression.Sample:
        return {
            "inputs": {"image": torch.ones(2, 2, 2)},
            "target": torch.full((1, 2, 2), torch.nan if index == 0 else 2.0),
        }


@pytest.mark.slow
@pytest.mark.parametrize(
    "task,source,stages,metric",
    [
        (segmentation, SparseLabels, SEGMENTER, "iou_macro"),
        (regression, SparseValues, REGRESSOR, "mae"),
    ],
)
def test_distributed_metrics_include_ranks_without_known_targets(
    task, source, stages, metric, tmp_path
):
    from datetime import timedelta

    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="forkserver", timeout=timedelta(seconds=15)),
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(
        task.Module(model_chain=stages),
        train_dataloaders=DataLoader(source(), batch_size=1, shuffle=False),
        val_dataloaders=DataLoader(source(), batch_size=1, shuffle=False),
    )

    assert torch.isfinite(trainer.callback_metrics[f"train_{metric}"])
    assert torch.isfinite(trainer.callback_metrics[f"val_{metric}"])
