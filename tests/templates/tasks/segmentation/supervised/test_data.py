from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from modules.segmentation import supervised
from geosave.model.spec import ModelSpec, Ref

FLIP = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]
NORMALIZE = {
    "image": [
        {"name": "Normalize", "init_args": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}}
    ]
}


def _spec(**changes: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["red", "nir"]}},
            "chips": {"size": 8},
            "inputs": {"image": Ref("optical")},
            **changes,
        }
    )


class Chips(Dataset[supervised.Sample]):
    """A dataset returning the segmentation `Sample` contract."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> supervised.Sample:
        return {
            "inputs": {"image": torch.full((2, 8, 8), float(index))},
            "target": torch.full((8, 8), index % 2),
        }


def _data(
    tmp_path: Path, spec: ModelSpec | None = None, **options: object
) -> supervised.DataModule:
    path = (spec or _spec()).save(tmp_path / "model_spec.yaml")
    return supervised.DataModule(
        spec=path, train=Chips(), val=Chips(), **{"num_workers": 0, **options}
    )


def test_a_datamodule_takes_a_spec_built_in_python() -> None:
    spec = _spec()

    data = supervised.DataModule(spec=spec, train=Chips(), val=Chips(), num_workers=0)

    assert data.spec == spec


def test_training_accepts_model_ready_inputs_without_validity():
    data = supervised.DataModule(
        _spec(), Chips(), Chips(), augmentations=FLIP, num_workers=0
    )
    _training(data, True)
    batch = _batch(times=3)

    result = data.on_after_batch_transfer(batch, 0)

    assert set(result) == {"inputs", "target"}
    torch.testing.assert_close(
        result["inputs"]["image"], batch["inputs"]["image"].flip(-1)
    )
    torch.testing.assert_close(result["target"], batch["target"].flip(-1))


def _batch(times: int = 0) -> supervised.Sample:
    shape = (2, times, 2, 8, 8) if times else (2, 2, 8, 8)
    image = torch.arange(float(np.prod(shape))).reshape(shape)
    return {
        "inputs": {"image": image, "location_coords": torch.ones(2, 2)},
        "target": torch.arange(2 * 8 * 8).reshape(2, 8, 8) % 3,
    }


def _training(data: supervised.DataModule, training: bool) -> supervised.DataModule:
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


def test_loaders_collate_samples_into_batches(tmp_path: Path) -> None:
    data = _data(tmp_path, batch_size=4)

    batch = next(iter(data.val_dataloader()))

    assert batch["inputs"]["image"].shape == (4, 2, 8, 8)
    assert batch["target"].shape == (4, 8, 8)
    assert batch["inputs"]["image"][:, 0, 0, 0].tolist() == [0, 1, 2, 3]


def test_the_test_split_needs_its_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test="):
        _data(tmp_path).test_dataloader()

    assert len(_data(tmp_path, test=Chips()).test_dataloader().dataset) == 6


def test_workers_start_without_this_process_state(tmp_path: Path) -> None:
    loader = _data(tmp_path, num_workers=2).train_dataloader()

    assert loader.multiprocessing_context.get_start_method() == "forkserver"
    assert loader.persistent_workers


def test_batch_augmentation_uses_default_crop_size(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, augmentations=[{"name": "RandomCrop"}]), True)

    batch = data.on_after_batch_transfer(_batch(), 0)

    assert batch["inputs"]["image"].shape[-2:] == (8, 8)
    assert batch["target"].shape[-2:] == (8, 8)


def test_augmentations_move_image_and_target_together_while_training(
    tmp_path: Path,
) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), True)
    original = _batch()

    batch = data.on_after_batch_transfer(_batch(), 0)

    image = original["inputs"]["image"]
    torch.testing.assert_close(batch["inputs"]["image"], image.flip(-1))
    assert torch.equal(batch["target"], original["target"].flip(-1))
    assert batch["target"].dtype == torch.int64
    assert torch.equal(batch["inputs"]["location_coords"], torch.ones(2, 2))


def test_cutmix_pastes_the_same_box_into_pixels_and_labels(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, cutmix=1.0), True)
    original = _batch(times=3)

    batch = data.on_after_batch_transfer(_batch(times=3), 0)

    # Every pixel value is unique, so the box is wherever pixels changed.
    changed = batch["inputs"]["image"] != original["inputs"]["image"]
    box = changed.flatten(1, 2).all(dim=1)
    target = original["target"]
    assert box.any()
    assert torch.equal(batch["target"], torch.where(box, target.roll(1, 0), target))


def test_augmentations_do_not_run_outside_training(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, augmentations=FLIP), False)
    original = _batch()

    batch = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(batch["inputs"]["image"], original["inputs"]["image"])
    assert torch.equal(batch["target"], original["target"])


@pytest.mark.parametrize("training", [True, False])
def test_transforms_run_in_every_split_after_augmentation(
    tmp_path: Path, training: bool
) -> None:
    spec = _spec(transforms=NORMALIZE)
    data = _training(_data(tmp_path, spec, augmentations=FLIP), training)
    image = _batch()["inputs"]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(), 0)["inputs"]["image"]

    expected = (image.flip(-1) if training else image) - mean
    torch.testing.assert_close(result, expected / std)


def test_input_transforms_preserve_nan_values(tmp_path: Path) -> None:
    data = _training(_data(tmp_path, _spec(transforms=NORMALIZE)), False)
    batch = _batch()
    batch["inputs"]["image"][0, :, :2] = torch.nan

    result = data.on_after_batch_transfer(batch, 0)["inputs"]["image"]

    assert result[0, :, :2].isnan().all()


def test_a_time_axis_survives_augmentation_and_transforms(tmp_path: Path) -> None:
    spec = _spec(transforms=NORMALIZE)
    data = _training(_data(tmp_path, spec, augmentations=FLIP), True)
    image = _batch(times=3)["inputs"]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(times=3), 0)["inputs"]["image"]

    assert result.shape == (2, 3, 2, 8, 8)
    torch.testing.assert_close(result, (image.flip(-1) - mean) / std)


def test_a_pixel_an_augmentation_invents_is_ignored(tmp_path: Path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(_data(tmp_path, augmentations=shift, ignore_index=9), True)
    batch = _batch()
    batch["target"] = torch.zeros(2, 8, 8, dtype=torch.int64)

    moved = data.on_after_batch_transfer(batch, 0)["target"]

    assert set(moved.unique().tolist()) == {0, 9}


def test_nonfinite_input_values_are_preserved(tmp_path: Path) -> None:
    data = _training(_data(tmp_path), False)
    batch = _batch(times=2)
    batch["inputs"]["image"][0, 1, 0, 3, 4] = torch.nan
    batch["inputs"]["image"][0, 1, 0, 3, 5] = torch.inf
    target = batch["target"].clone()

    result = data.on_after_batch_transfer(batch, 0)

    assert result["inputs"]["image"][0, 1, 0, 3, 4].isnan()
    assert result["inputs"]["image"][0, 1, 0, 3, 5] == torch.inf
    torch.testing.assert_close(result["target"], target)


def test_multiple_pixel_inputs_share_geometry_and_preserve_context(
    tmp_path: Path,
) -> None:
    spec = _spec(inputs={"image": Ref("optical"), "dem": Ref("optical")})
    crop = [{"name": "CenterCrop", "init_args": {"size": (4, 4)}}]
    data = _training(_data(tmp_path, spec, augmentations=crop), True)
    batch = _batch(times=2)
    batch["inputs"]["dem"] = torch.arange(2 * 8 * 8, dtype=torch.float32).reshape(
        2, 1, 8, 8
    )

    result = data.on_after_batch_transfer(batch, 0)

    for name in ("image", "dem"):
        torch.testing.assert_close(
            result["inputs"][name], batch["inputs"][name][..., 2:6, 2:6]
        )
    torch.testing.assert_close(result["target"], batch["target"][..., 2:6, 2:6])
    assert result["inputs"]["location_coords"] is batch["inputs"]["location_coords"]


def test_lightning_cli_builds_each_split_dataset(tmp_path: Path) -> None:
    from geosave.ml.cli import GeosaveCLI

    chips = {"class_path": f"{__name__}.Chips"}
    cli = GeosaveCLI(
        supervised.Module,
        supervised.DataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=False,
        auto_configure_optimizers=False,
        args={
            "model": {"model_chain": {"model": {"class_path": "torch.nn.Identity"}}},
            "data": {
                "spec": str(_spec().save(tmp_path / "model_spec.yaml")),
                "train": chips,
                "val": chips,
                "num_workers": 0,
            },
            "trainer": {"logger": False, "default_root_dir": str(tmp_path)},
        },
    )

    assert isinstance(cli.datamodule.train_dataset, Chips)
    assert cli.datamodule.test_dataset is None
