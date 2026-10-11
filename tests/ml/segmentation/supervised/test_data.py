from __future__ import annotations

import gc
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import orjson
import pytest
import torch
import xarray as xr
from lightning import LightningDataModule
from odc.geo.geobox import GeoBox
from torch.utils.data import DataLoader

from geosave_engine.geodata import GeoVector, raster, open_vector, stack
from geosave_engine.geodata.stac import create_items
from geosave_engine.ml.segmentation import supervised
from geosave_engine.model.spec import ModelSpec, Ref
from tests.geodata.conftest import stack_templates


_spec_processing_not_settled = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="ModelSpec processing is a stub until frames, chips, inputs and context are redesigned",
)

SIZE = 20


def test_batch_augmentation_uses_default_crop_size(tmp_path):
    data = _training(_batches(tmp_path, augmentations=[{"name": "RandomCrop"}]), True)
    image, target, _, _ = data.on_after_batch_transfer(_batch(), 0)
    assert image["image"].shape[-2:] == (8, 8)
    assert target.shape[-2:] == (8, 8)


def double(data: xr.Dataset) -> xr.Dataset:
    """Stand in for a preprocessing step that keeps the grid."""
    return data * 2


def location_context(raster):
    from geosave_engine.model.encoder.prithvi import location_coords

    return {"location_coords": location_coords(raster)}


def _pixels(index: int, times: int = 0) -> np.ndarray:
    plane = np.arange(SIZE * SIZE, dtype="float32").reshape(SIZE, SIZE) + 1000 * index
    return np.stack([plane + day for day in range(times)]) if times else plane


def _samples(*, times: int = 0, label: bool = True) -> list[dict[str, xr.Dataset]]:
    """Two samples side by side, each an optical raster and, unless left out, a label."""
    samples = []
    for index in range(2):
        left = 500_000 + index * 1000
        grid = GeoBox.from_bbox(
            (left, 9_000_000, left + SIZE * 10, 9_000_000 + SIZE * 10),
            "EPSG:32748",
            resolution=10,
        )
        days = np.array(
            [f"2025-06-{day + 1:02d}" for day in range(times)], "datetime64[ns]"
        )
        dims = ("time", "y", "x") if times else ("y", "x")
        groups = {
            "optical": raster(
                {
                    "red": (dims, _pixels(index, times)),
                    "nir": (dims, _pixels(index, times) + 0.5),
                },
                grid,
                **({"time": days} if times else {}),
            )
        }
        if label or index == 0:
            groups["label"] = raster(
                {"class": (("y", "x"), np.full((SIZE, SIZE), index, "uint8"))},
                grid,
            )
        samples.append(groups)
    return samples


def _catalog(root: Path, samples: list[dict[str, xr.Dataset]]) -> gpd.GeoDataFrame:
    """Write each sample as COGs and return the item table listing them."""
    items = []
    for index, groups in enumerate(samples):
        sample = stack(groups)
        files = sample.gs.to_cog(stack_templates(sample, root / f"s{index}"))
        # Timeless samples are dated by hand.
        dated = any("time" in raster.dims for raster in groups.values())
        when = None if dated else datetime(2025, 6, 1, tzinfo=UTC)
        items.append(create_items(files, id=f"s{index}", datetime=when))
    path = GeoVector.concat(items).gs.to_geoparquet(root / "catalog.parquet")
    # Building read each sample lazily; collect, so no test starts with them open.
    gc.collect()
    return open_vector(path)


def _manifest(root: Path, *, times: int = 0, label: bool = True) -> gpd.GeoDataFrame:
    """Write two raw samples and return the manifest listing their groups."""
    return _catalog(root, _samples(times=times, label=label))


def _prepared(
    root: Path, spec: ModelSpec | None = None, **options: object
) -> gpd.GeoDataFrame:
    """Write two samples as the spec's model inputs beside their labels."""
    spec = spec or _spec()
    prepared = []
    for groups in _samples(**options):
        results = spec.preprocess(groups)
        inputs = {name: results[name] for name in spec.input_rasters}
        prepared.append({**inputs, **{k: v for k, v in groups.items() if k == "label"}})
    rows = _catalog(root / "prepared", prepared)
    # A prepared catalog states the spec that made it.
    spec.save(root / "prepared" / "model_spec.yaml")
    return rows


def _spec(**changes: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["red", "nir"]}},
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.double",
                    "kwargs": {"data": Ref("optical")},
                }
            },
            "chips": {"size": 8, "overlap": 2, "window": "hann"},
            "inputs": {
                "image": Ref("image"),
            },
            "context": {
                "call": f"{__name__}.location_context",
                "kwargs": {"raster": Ref("image")},
            },
            **changes,
        }
    )


@_spec_processing_not_settled
def test_an_input_bound_to_a_raster_reads_the_variables_its_spec_selects(
    tmp_path: Path,
) -> None:
    spec = _spec(
        rasters={"optical": {"variables": ["nir"]}},
        preprocessing={},
        inputs={"image": Ref("optical")},
        context=None,
    )

    model_inputs, _, _, _ = supervised.Dataset(_prepared(tmp_path, spec), spec)[0]

    # The samples hold red and nir; nir alone ends in a half.
    assert model_inputs["image"].shape == (1, 8, 8)
    assert torch.all(model_inputs["image"] % 1 == 0.5)


@_spec_processing_not_settled
def test_a_dataset_numbers_every_tile_of_every_sample(tmp_path: Path) -> None:
    dataset = supervised.Dataset(_prepared(tmp_path), _spec())

    # A 20-pixel side takes four 8-pixel tiles stepping by 6, so 16 per sample.
    assert len(dataset) == 32
    assert dataset.windows["parent"].value_counts().tolist() == [16, 16]


@_spec_processing_not_settled
def test_a_sample_holds_the_declared_inputs_a_target_and_its_number(
    tmp_path: Path,
) -> None:
    dataset = supervised.Dataset(_prepared(tmp_path), _spec())

    model_inputs, target, index, valid = dataset[21]

    assert index == 21
    assert valid.shape == target.shape
    assert sorted(model_inputs) == ["image", "location_coords"]
    assert model_inputs["image"].shape == (2, 8, 8)
    assert model_inputs["image"].dtype == torch.float32
    assert model_inputs["location_coords"].shape == (2,)
    assert target.shape == (8, 8)
    assert target.dtype == torch.int64
    assert (target == 1).all()  # tile 21 belongs to the second sample
    # Tile 5 of a sample starts 3 pixels in, past the frame, on both axes.
    expected = 2 * _pixels(1)[3:11, 3:11]
    np.testing.assert_array_equal(model_inputs["image"][0].numpy(), expected)


@_spec_processing_not_settled
def test_each_tile_reads_its_own_location(tmp_path: Path) -> None:
    dataset = supervised.Dataset(_prepared(tmp_path), _spec())

    first = dataset[0][0]["location_coords"]
    last = dataset[15][0]["location_coords"]

    assert first[0] > last[0]  # rows run south
    assert first[1] < last[1]  # columns run east


@_spec_processing_not_settled
def test_a_target_pixel_the_raster_lacks_is_ignored(tmp_path: Path) -> None:
    spec = _spec(chips={"size": 8, "overlap": 2, "mode": "constant"})
    dataset = supervised.Dataset(_prepared(tmp_path, spec), spec, ignore_index=7)

    model_inputs, target, _, _ = dataset[0]

    assert (target[:3] == 7).all() and (target[3:, 3:] == 0).all()
    assert model_inputs["image"][:, :3].isnan().all()


@_spec_processing_not_settled
def test_a_row_lacking_a_layer_is_refused_by_name(tmp_path: Path) -> None:
    manifest = _prepared(tmp_path)
    manifest = manifest[manifest["id"] != "s1/image"]

    with pytest.raises(ValueError, match=r"s1/label.*\['image'\]"):
        supervised.Dataset(manifest, _spec())


@_spec_processing_not_settled
def test_building_a_dataset_computes_no_pixels(tmp_path: Path) -> None:
    import dask

    def refuse(*args, **kwargs):
        raise AssertionError("the dataset computed pixels")

    manifest = _prepared(tmp_path)

    with dask.config.set(scheduler=refuse):
        dataset = supervised.Dataset(manifest, _spec())

    assert len(dataset) == 32


@_spec_processing_not_settled
def test_a_spec_without_chips_or_inputs_is_refused(tmp_path: Path) -> None:
    manifest = _prepared(tmp_path)

    with pytest.raises(ValueError, match="chips"):
        supervised.Dataset(manifest, _spec(chips=None))
    with pytest.raises(ValueError, match="inputs"):
        supervised.Dataset(manifest, _spec(inputs={}))


@_spec_processing_not_settled
def test_merged_tiles_rebuild_every_sample(tmp_path: Path) -> None:
    from tiler import Merger

    dataset = supervised.Dataset(_prepared(tmp_path), _spec())
    layout, halo = dataset.spec.chips.layout((SIZE, SIZE))
    mergers = [
        Merger(layout, logits=2, window="hann", save_visits=False)
        for _ in range(len(dataset.frames))
    ]
    for number in range(len(dataset)):
        model_inputs, _, position, _ = dataset[number]
        row = dataset.windows.iloc[position]
        mergers[int(row.parent)].add(int(row.chip), model_inputs["image"].numpy())
    for index in range(2):
        rebuilt = mergers[index].merge(extra_padding=halo)
        np.testing.assert_allclose(rebuilt[0], 2 * _pixels(index), rtol=1e-5)
        parent = dataset.frames.iloc[index].gs.to_stack()
        labels = parent[dataset.target].dataset.gs.to_tensor()
        assert labels.shape == (1, SIZE, SIZE)
        assert (labels == index).all()


@_spec_processing_not_settled
def test_frames_put_a_time_axis_ahead_of_the_bands(tmp_path: Path) -> None:
    spec = _spec(frames={"length": 2, "tolerance": "1D"})
    dataset = supervised.Dataset(_prepared(tmp_path, spec, times=4), spec)

    model_inputs, target, _, _ = dataset[16]

    # Four instants make two frames per sample; tile 16 opens the second frame.
    assert len(dataset) == 2 * 2 * 16
    assert model_inputs["image"].shape == (2, 2, 8, 8)
    assert target.shape == (8, 8)
    np.testing.assert_array_equal(
        model_inputs["image"][:, 0, 3, 3].numpy(), 2 * _pixels(0, 4)[2:, 0, 0]
    )


def _open_rasters() -> list[str]:
    links = [Path("/proc/self/fd") / name for name in os.listdir("/proc/self/fd")]
    return [str(link.resolve()) for link in links if link.resolve().suffix == ".tif"]


@_spec_processing_not_settled
@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="needs /proc")
def test_a_built_dataset_leaves_no_raster_open(tmp_path: Path) -> None:
    manifest = _prepared(tmp_path)
    before = _open_rasters()

    dataset = supervised.Dataset(manifest, _spec())

    assert len(dataset) == 32
    assert _open_rasters() == before


@pytest.mark.slow
def test_worker_processes_read_their_own_rasters(tmp_path: Path) -> None:
    # A worker started fresh imports only installed code, so no test-local call.
    library_call = {
        "image": {
            "call": "geosave_engine.geodata.transform.nodata.to_nan",
            "kwargs": {"data": Ref("optical")},
        }
    }
    spec = _spec(preprocessing=library_call, context=None)
    dataset = supervised.Dataset(_prepared(tmp_path, spec), spec)
    dataset[0]  # this process has read pixels, which a forked worker would hang on

    loader = DataLoader(
        dataset, batch_size=8, num_workers=2, multiprocessing_context="forkserver"
    )
    batches = list(loader)

    numbers = [int(key) for _, _, keys, _ in batches for key in keys]
    images = torch.cat([model_inputs["image"] for model_inputs, _, _, _ in batches])
    assert sorted(numbers) == list(range(len(dataset)))
    for number, image in zip(numbers, images, strict=True):
        assert torch.equal(image, dataset[number][0]["image"])


def _data(tmp_path: Path, spec: ModelSpec | None = None, **options: object):
    _prepared(tmp_path / "train", spec)
    _prepared(tmp_path / "val", spec)
    path = (spec or _spec()).save(tmp_path / "model_spec.yaml")
    return supervised.DataModule(
        spec=path,
        train=tmp_path / "train/prepared/catalog.parquet",
        val=tmp_path / "val/prepared/catalog.parquet",
        **{"batch_size": 4, "num_workers": 0, **options},
    )


def _batches(
    tmp_path: Path, spec: ModelSpec | None = None, **options: object
) -> supervised.DataModule:
    """Build only the batch pipelines, so batch tests read no catalog."""
    path = (spec or _spec()).save(tmp_path / "model_spec.yaml")
    data = supervised.DataModule(spec=path, train=tmp_path, val=tmp_path, **options)
    data.setup("predict")
    return data


def _batch(
    times: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    shape = (2, times, 2, 8, 8) if times else (2, 2, 8, 8)
    image = torch.arange(float(np.prod(shape))).reshape(shape)
    target = torch.arange(2 * 8 * 8).reshape(2, 8, 8) % 3
    return (
        {"image": image, "location_coords": torch.ones(2, 2)},
        target,
        torch.tensor([4, 9]),
        torch.ones((2, 8, 8), dtype=torch.bool),
    )


def _training(data: supervised.DataModule, training: bool) -> supervised.DataModule:
    data.trainer = SimpleNamespace(training=training)  # type: ignore[assignment]
    return data


@_spec_processing_not_settled
def test_setup_reads_each_split_from_its_own_manifest(tmp_path: Path) -> None:
    data = _data(tmp_path)

    data.setup("fit")
    model_inputs, target, index, _ = next(iter(data.train_dataloader()))

    assert isinstance(data, LightningDataModule)
    assert len(data.train_dataset) == len(data.val_dataset) == 32
    assert model_inputs["image"].shape == (4, 2, 8, 8)
    assert model_inputs["location_coords"].shape == (4, 2)
    assert target.shape == (4, 8, 8)
    assert len(index) == 4
    assert next(iter(data.val_dataloader()))[2].tolist() == [0, 1, 2, 3]


@_spec_processing_not_settled
def test_the_test_split_needs_its_manifest(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="test="):
        data.setup("test")

    data = supervised.DataModule(
        spec=tmp_path / "model_spec.yaml",
        train=tmp_path / "train/prepared/catalog.parquet",
        val=tmp_path / "val/prepared/catalog.parquet",
        test=tmp_path / "val/prepared/catalog.parquet",
        num_workers=0,
    )
    data.setup("test")
    assert len(data.test_dataloader().dataset) == 32


@_spec_processing_not_settled
def test_workers_start_without_this_process_state(tmp_path: Path) -> None:
    data = _data(tmp_path, num_workers=2)
    data.setup("fit")

    loader = data.train_dataloader()

    assert loader.multiprocessing_context.get_start_method() == "forkserver"
    assert loader.persistent_workers


FLIP = [{"name": "RandomHorizontalFlip", "init_args": {"p": 1.0}}]
NORMALIZE = {
    "image": [
        {"name": "Normalize", "init_args": {"mean": [1.0, 2.0], "std": [2.0, 4.0]}}
    ]
}


def test_augmentations_move_image_and_target_together_while_training(
    tmp_path: Path,
) -> None:
    data = _training(_batches(tmp_path, augmentations=FLIP), True)
    model_inputs, target, index, valid = _batch()

    augmented, moved, numbers, _ = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(augmented["image"], model_inputs["image"].flip(-1))
    assert torch.equal(moved, target.flip(-1))
    assert moved.dtype == torch.int64
    assert torch.equal(augmented["location_coords"], model_inputs["location_coords"])
    assert torch.equal(numbers, index)


def test_cutmix_pastes_the_same_box_into_pixels_and_labels(tmp_path: Path) -> None:
    data = _training(_batches(tmp_path, cutmix=1.0), True)
    model_inputs, target, _, _ = _batch(times=3)

    mixed, labels, _, _ = data.on_after_batch_transfer(_batch(times=3), 0)

    # Every pixel value is unique, so the box is wherever pixels changed.
    box = (mixed["image"] != model_inputs["image"]).flatten(1, 2).all(dim=1)
    assert box.any()
    assert torch.equal(labels, torch.where(box, target.roll(1, 0), target))


def test_augmentations_do_not_run_outside_training(tmp_path: Path) -> None:
    data = _training(_batches(tmp_path, augmentations=FLIP), False)
    model_inputs, target, _, _ = _batch()

    kept, same, _, _ = data.on_after_batch_transfer(_batch(), 0)

    torch.testing.assert_close(kept["image"], model_inputs["image"])
    assert torch.equal(same, target)


@pytest.mark.parametrize("training", [True, False])
def test_transforms_run_in_every_split_after_augmentation(
    tmp_path: Path, training: bool
) -> None:
    spec = _spec(transforms=NORMALIZE)
    data = _training(_batches(tmp_path, spec, augmentations=FLIP), training)
    image = _batch()[0]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(), 0)[0]["image"]

    expected = (image.flip(-1) if training else image) - mean
    torch.testing.assert_close(result, expected / std)


def test_pixels_a_raster_lacks_become_zero_after_transforms(tmp_path: Path) -> None:
    data = _training(_batches(tmp_path, _spec(transforms=NORMALIZE)), False)
    model_inputs, target, index, valid = _batch()
    model_inputs["image"][0, :, :2] = torch.nan

    result = data.on_after_batch_transfer((model_inputs, target, index, valid), 0)[0][
        "image"
    ]

    assert not result.isnan().any()
    assert (result[0, :, :2] == 0).all()


def test_a_time_axis_survives_augmentation_and_transforms(tmp_path: Path) -> None:
    spec = _spec(transforms=NORMALIZE)
    data = _training(_batches(tmp_path, spec, augmentations=FLIP), True)
    image = _batch(times=3)[0]["image"]
    mean = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1, 1)
    std = torch.tensor([2.0, 4.0]).reshape(1, 1, 2, 1, 1)

    result = data.on_after_batch_transfer(_batch(times=3), 0)[0]["image"]

    assert result.shape == (2, 3, 2, 8, 8)
    torch.testing.assert_close(result, (image.flip(-1) - mean) / std)


def test_a_pixel_an_augmentation_invents_is_ignored(tmp_path: Path) -> None:
    shift = [
        {
            "name": "RandomAffine",
            "init_args": {"degrees": 0.0, "translate": [0.5, 0.5], "p": 1.0},
        }
    ]
    data = _training(_batches(tmp_path, augmentations=shift, ignore_index=9), True)
    target = torch.zeros(2, 8, 8, dtype=torch.int64)

    moved = data.on_after_batch_transfer(
        (_batch()[0], target, _batch()[2], _batch()[3]), 0
    )[1]

    assert set(moved.unique().tolist()) == {0, 9}


@_spec_processing_not_settled
def test_frames_and_reopened_workers_read_the_same_chips(tmp_path):
    import pickle

    spec = _spec(frames={"length": 2, "stride": 1, "tolerance": "1D"})
    manifest = _prepared(tmp_path, spec, times=4)
    dataset = supervised.Dataset(manifest, spec)
    # Four instants, two at a time, stepping by one: three frames per sample.
    assert dataset.frames["parent"].tolist() == [0, 0, 0, 1, 1, 1]
    reopened = pickle.loads(pickle.dumps(dataset))
    assert reopened[17][2] == 17
    torch.testing.assert_close(reopened[17][0]["image"], dataset[17][0]["image"])
    reordered = supervised.Dataset(manifest.iloc[::-1], spec)
    assert len(reordered) == len(dataset)
    # Same spatial grids, different temporal windows remain distinct frames.
    first, second = (dataset.frames.iloc[i].gs.to_stack() for i in (0, 1))
    assert first.gs.geobox == second.gs.geobox


def test_validity_survives_nan_conversion_without_changing_targets(tmp_path):
    data = _training(_batches(tmp_path), False)
    inputs, target, ids, valid = _batch(times=2)
    inputs["image"][0, 1, 0, 3, 4] = torch.nan
    converted, labels, keys, mask = data.on_after_batch_transfer(
        (inputs, target, ids, valid), 0
    )
    assert converted["image"][0, 1, 0, 3, 4] == 0
    assert not mask[0, 3, 4]
    assert mask[1].all()
    assert torch.equal(keys, ids)
    torch.testing.assert_close(labels, target)


@_spec_processing_not_settled
@pytest.mark.parametrize(
    "frames", [None, {"length": 2, "stride": 1, "tolerance": "1D"}]
)
def test_a_chip_carries_its_anchors_properties_and_reads_its_frame(tmp_path, frames):
    manifest = _prepared(tmp_path, _spec(frames=frames), times=4 if frames else 0)
    manifest["class_id"] = manifest["id"].str.split("/").str[0].map({"s0": 7, "s1": 9})
    dataset = supervised.Dataset(manifest, _spec(frames=frames))
    # The second sample's chip 5; with frames, of its second frame.
    position = (4 if frames else 1) * 16 + 5
    row = dataset.windows.iloc[position]
    assert row.class_id == 9
    sources = orjson.loads(row.sources)
    assert set(sources) == {"image", "label"}
    tile = row.gs.to_stack()
    assert tile.gs.rasters["label"].gs.to_tensor().shape == (1, 8, 8)
    expected = (
        2 * _pixels(1, 4)[1:3, 3:11, 3:11] if frames else 2 * _pixels(1)[3:11, 3:11]
    )
    np.testing.assert_array_equal(tile.gs.rasters["image"].red, expected)
    if frames:
        days = [item["properties"]["datetime"][:10] for item in sources["image"]]
        assert days == ["2025-06-02", "2025-06-03"]


def test_evaluation_validity_intersects_named_inputs_and_time_axes(tmp_path):
    spec = _spec(inputs={"image": Ref("optical"), "dem": Ref("optical")})
    data = _training(_batches(tmp_path, spec), False)
    inputs, target, ids, valid = _batch(times=2)
    inputs["dem"] = torch.ones((2, 1, 8, 8))
    inputs["image"][0, 1, 1, 3, 4] = torch.nan
    inputs["dem"][1, 0, 4, 5] = torch.inf
    _, kept, _, mask = data.on_after_batch_transfer((inputs, target, ids, valid), 0)
    assert not mask[0, 3, 4] and not mask[1, 4, 5]
    assert mask[0, 4, 5] and mask[1, 3, 4]
    torch.testing.assert_close(kept, target)


@_spec_processing_not_settled
def test_an_empty_manifest_is_refused(tmp_path: Path) -> None:
    manifest = _prepared(tmp_path)

    with pytest.raises(ValueError, match="no rows"):
        supervised.Dataset(manifest.iloc[0:0], _spec())


@_spec_processing_not_settled
def test_a_chip_opens_only_its_own_sample(tmp_path: Path, monkeypatch) -> None:
    from geosave_engine.geodata.io.raster import cogs

    dataset = supervised.Dataset(_prepared(tmp_path), _spec())
    opened = set()
    read = cogs.read
    monkeypatch.setattr(
        cogs,
        "read",
        lambda files, **options: (
            opened.update(
                part
                for file in files
                for part in Path(file.href).parts
                if part in ("s0", "s1")
            )
            or read(files, **options)
        ),
    )

    dataset[0]
    dataset[1]

    assert opened == {"s0"}
