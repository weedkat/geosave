"""YAML processing from native rasters through model outputs."""

from pathlib import Path

import dask.array as da
from dask.callbacks import Callback
import numpy as np
from odc.geo.geobox import GeoBox
import pytest
import torch
import xarray as xr

import geosave as gs
from geosave.model.spec import ModelSpec


RECIPE = (
    Path(__file__).resolve().parents[3] / "docs/guides/model_spec.yaml"
).read_text()


def load_recipe(tmp_path: Path, text: str = RECIPE) -> ModelSpec:
    path = tmp_path / "recipe.yaml"
    path.write_text(text)
    return ModelSpec.load(path)


def scene() -> xr.Dataset:
    grid = GeoBox.from_bbox(
        (500000, 9000000, 500080, 9000080), "EPSG:32748", resolution=10
    )
    red = np.arange(128, dtype="uint16").reshape(2, 8, 8) + 100
    data = xr.Dataset(
        {
            "nir": (("time", "y", "x"), da.from_array(red + 200, chunks=(1, 4, 4))),
            "red": (("time", "y", "x"), da.from_array(red, chunks=(1, 4, 4))),
        },
        coords={"time": np.array(["2025-06-01", "2025-06-02"], dtype="datetime64[ns]")},
    ).gs.write_grid(grid)
    for variable in data.data_vars.values():
        variable.attrs.update(scale_factor=0.01, add_offset=0.0, _FillValue=65535)
    return data


def test_yaml_preserves_lazy_preparation_and_explicit_temporal_tensor_order(tmp_path):
    spec = load_recipe(tmp_path)
    raw = scene()
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(1)):
        prepared = spec.preprocess({"optical": raw})
    assert tasks == []
    assert isinstance(prepared["reflectance"].red.data, da.Array)

    encoded = spec.encode(prepared)
    inputs = spec.transform(spec.model_inputs(encoded))
    expected = np.stack([raw.red.values, raw.nir.values]).astype("float32") * 0.01
    np.testing.assert_allclose(
        inputs["image"].numpy(), (expected - expected.mean()) / 2, atol=1e-6
    )
    assert tuple(inputs["image"].shape) == (2, 2, 8, 8)
    assert "center" not in inputs
    assert encoded["optical"] is prepared["optical"]

    reloaded = ModelSpec.load(spec.save(tmp_path / "saved.yaml"))
    restored = reloaded.encode(reloaded.preprocess({"optical": raw}))
    torch.testing.assert_close(
        reloaded.transform(reloaded.model_inputs(restored))["image"], inputs["image"]
    )


def test_two_samples_keep_their_own_inverse_state(tmp_path):
    spec = load_recipe(tmp_path)
    first = scene().isel(x=slice(0, 4), y=slice(0, 4))
    second = scene().isel(x=slice(4, 8), y=slice(4, 8))
    states = []
    for raster in (first, second):
        states.append(spec.encode(spec.preprocess({"optical": raster})))
    for raster, values in zip((first, second), states, strict=True):
        image = spec.transform(spec.model_inputs(values))["image"]
        prediction = image.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
        result = spec.decode(prediction, values)["raster"]
        expected = (raster.red.values + raster.nir.values).mean(axis=0) * 0.005
        np.testing.assert_allclose(result.values[0], expected, atol=1e-6)
        assert result.gs.geobox == raster.gs.geobox


def test_model_boundary_rejects_wrong_channel_count_and_dtype(tmp_path):
    spec = load_recipe(tmp_path)
    with pytest.raises(ValueError, match="image.*shape"):
        spec.transform({"image": torch.zeros(3, 2, 4, 4)})
    with pytest.raises(ValueError, match="image.*dtype"):
        spec.transform({"image": torch.zeros(2, 2, 4, 4, dtype=torch.float64)})


def test_invalid_encoding_reference_fails_before_any_call(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_recipe(
            tmp_path, RECIPE.replace("raster: !ref reflectance", "raster: !ref missing")
        )


def test_postprocessing_selects_classes_after_merging_logits(tmp_path):
    spec = load_recipe(
        tmp_path,
        """schema_version: 3
rasters: {}
postprocessing:
  classes:
    call: !ref prediction.argmax
    kwargs: {dim: band}
""",
    )
    raw = scene().isel(time=0, drop=True)
    tree = raw.gs.to_datatree("optical")
    chips = gs.Chips([tree], size=4, resolution=10, overlap=2)
    for index in range(len(chips)):
        # Class IDs [0, 1, 2] cannot be meaningfully averaged.
        logits = np.zeros((3, 4, 4), dtype="float32")
        logits[0] = 2
        logits[2] = 1 if index % 2 else 5
        chips.add(index, logits)
    (merged,) = chips.merge()
    result = spec.postprocess(merged)["classes"]
    np.testing.assert_array_equal(result.values, merged.values.argmax(axis=0))
    assert set(np.unique(result)) <= {0, 2}
    assert result.gs.geobox == merged.gs.geobox


def test_catalog_training_checkpoint_and_prediction_round_trip(tmp_path):
    from lightning.pytorch import Trainer
    from torch.utils.data import Dataset
    from modules.regression import supervised

    spec = load_recipe(tmp_path)
    raw = scene()
    catalog = raw.gs.to_catalog(tmp_path / "assets", collection="optical")
    catalog = gs.open_vector(catalog.gs.to_geoparquet(tmp_path / "catalog.parquet"))
    samples = catalog.gs.sample(pair=["optical"], window=2)
    assert len(samples) == 1
    tree = samples[0].gs.to_datatree()
    try:
        tasks = []
        with Callback(posttask=lambda *args: tasks.append(1)):
            chips = gs.Chips([tree], size=4, resolution=10)
            _, chip = chips[0]
            prepared = spec.preprocess({"optical": chip["optical"].gs.to_dataset()})
        assert tasks == []
        assert isinstance(prepared["reflectance"].red.data, da.Array)

        class TrainingSamples(Dataset):
            def __len__(self):
                return len(chips)

            def __getitem__(self, index):
                _, chip = chips[index]
                values = spec.encode(
                    spec.preprocess({"optical": chip["optical"].gs.to_dataset()})
                )
                inputs = spec.model_inputs(values)
                # An independent target in normalized output space.
                target = inputs["image"].mean(dim=(0, 1)).unsqueeze(0)
                return {"inputs": inputs, "target": target}

        data = supervised.DataModule(
            spec,
            TrainingSamples(),
            TrainingSamples(),
            size=4,
            batch_size=2,
            num_workers=0,
        )
        module = supervised.Module(
            model_chain={"model": {"class_path": f"{__name__}.TemporalRegressor"}},
            optimizer={"name": "sgd", "init_args": {"lr": 0.1}},
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_path,
        )
        trainer.fit(module, datamodule=data)
        assert trainer.global_step == 2
        assert torch.isfinite(trainer.callback_metrics["val_loss"])
        assert module.model.get_submodule("model").factor.item() != 1

        checkpoint = tmp_path / "trained.ckpt"
        trainer.save_checkpoint(checkpoint)
        recipe = spec.save(tmp_path / "model_spec.yaml")
        restored = supervised.Module.load_from_checkpoint(
            checkpoint, weights_only=False
        ).eval()
        inference = ModelSpec.load(recipe)
        # The generated DataModule's evaluation path and inference use exactly
        # the same transforms, including their batching convention.
        batch = next(iter(data.val_dataloader()))
        trained_inputs = data.on_after_batch_transfer(batch, 0)["inputs"]
        inferred_inputs = inference.transform(batch["inputs"], batched=True)
        torch.testing.assert_close(trained_inputs["image"], inferred_inputs["image"])

        expected = np.empty((8, 8), dtype="float32")
        factor = restored.model.get_submodule("model").factor.detach().item()
        for index in range(len(chips)):
            _, chip = chips[index]
            raster = chip["optical"].gs.to_dataset()
            values = inference.encode(inference.preprocess({"optical": raster}))
            inputs = inference.transform(inference.model_inputs(values))
            with torch.no_grad():
                prediction = restored(image=inputs["image"].unsqueeze(0))[0]
            output = inference.decode(prediction, values)["raster"]
            chips.add(index, output.values)
            # Compare with ordinary NumPy, independent of recipe execution.
            pixels = (
                np.stack([raster.red.values, raster.nir.values]).astype("float32")
                * 0.01
            )
            center = pixels.mean()
            answer = ((pixels - center) / 2).mean(axis=(0, 1)) * factor * 2 + center
            grid = chips.grid(index)
            row = round((raw.gs.geobox.transform.f - grid.transform.f) / 10)
            col = round((grid.transform.c - raw.gs.geobox.transform.c) / 10)
            expected[row : row + 4, col : col + 4] = answer
        (merged,) = chips.merge()
        np.testing.assert_allclose(merged.values[0], expected, atol=1e-6)
        assert merged.gs.geobox == raw.gs.geobox
        output = merged.isel(band=0, drop=True).rename("prediction").to_dataset()
        predictions = output.gs.to_catalog(
            tmp_path / "predictions", collection="prediction", datetime=tree.gs.timespan
        )
        entry = predictions.gs.to_items()[0]
        assert entry.datetime is None
        assert entry.common_metadata.start_datetime.date().isoformat() == "2025-06-01"
        assert entry.common_metadata.end_datetime.date().isoformat() == "2025-06-02"
        saved = predictions.gs.to_geoparquet(tmp_path / "predictions.parquet")
        with gs.open_vector(saved).gs.to_datatree() as reopened:
            actual = reopened["prediction"].gs.to_dataset()
            np.testing.assert_allclose(
                actual.prediction.values.squeeze(), expected, atol=1e-6
            )
            assert actual.gs.geobox == raw.gs.geobox
    finally:
        tree.close()


class TemporalRegressor(torch.nn.Module):
    """Tiny real trainable model for the temporal processing smoke."""

    def __init__(self):
        super().__init__()
        self.variables = ["prediction"]
        self.factor = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(1, 2)).unsqueeze(1) * self.factor


def test_prepared_catalog_retains_rasters_needed_to_decode(tmp_path):
    spec = load_recipe(tmp_path)
    raw = scene().gs.to_catalog(tmp_path / "raw", collection="optical")
    prepared = spec.prepare(raw, tmp_path / "prepared")
    prepared = gs.open_vector(prepared.gs.to_geoparquet(tmp_path / "prepared.parquet"))
    assert set(prepared["collection"]) == {"reflectance", "optical"}
    sample = prepared.gs.sample(pair=["reflectance", "optical"], window=2)[0]
    with sample.gs.to_datatree() as tree:
        values = {name: tree[name].gs.to_dataset() for name in tree.children}
        values = spec.encode(values)
        image = spec.transform(spec.model_inputs(values))["image"]
        prediction = image.mean(dim=(0, 1)).unsqueeze(0)
        result = spec.decode(prediction, values)["raster"]
        raw = scene()
        expected = (raw.red.values + raw.nir.values).mean(axis=0) * 0.005
        np.testing.assert_allclose(result.values[0], expected, atol=1e-6)
        assert result.gs.geobox == raw.gs.geobox


def test_named_tensor_axes_keep_temporal_context_and_batches_consistent():
    from geosave.model.spec import Ref

    spec = ModelSpec.model_validate(
        {
            "schema_version": 3,
            "rasters": {"optical": {"channels": 1}},
            "inputs": {"image": Ref("optical"), "dates": Ref("optical")},
            "tensors": {
                "image": {"shape": ["time", 2, "height", "width"], "dtype": "float32"},
                "dates": {"shape": ["time", 2], "dtype": "float32"},
            },
        }
    )
    values = {"image": torch.zeros(3, 2, 2, 4, 4), "dates": torch.zeros(3, 2, 2)}
    spec.transform(values, batched=True)
    with pytest.raises(ValueError, match="dates.*shape"):
        spec.transform({**values, "dates": torch.zeros(3, 1, 2)}, batched=True)
    with pytest.raises(ValueError, match="dates.*batch"):
        spec.transform({**values, "dates": torch.zeros(2, 2, 2)}, batched=True)


def test_mixed_resolution_inputs_align_explicitly_before_encoding(tmp_path):
    optical = scene().isel(time=0, drop=True)
    grid = GeoBox.from_bbox(
        (500000, 9000000, 500080, 9000080), "EPSG:32748", resolution=20
    )
    dem = xr.Dataset(
        {"height": (("y", "x"), da.full((4, 4), 7, chunks=(2, 2), dtype="float32"))}
    ).gs.write_grid(grid)
    spec = load_recipe(
        tmp_path,
        """schema_version: 3
rasters:
  optical: {variables: [red, nir], require_crs: true}
  dem: {variables: [height], require_crs: true}
preprocessing:
  reflectance:
    call: !ref optical.gs.mask_and_scale
  elevation:
    call: !ref dem.gs.reproject
    kwargs: {target: !ref optical.gs.geobox, resampling: nearest}
encoding:
  image:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref reflectance, dims: [band, y, x]}
  elevation:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref elevation, dims: [band, y, x]}
inputs:
  image: !ref image
  elevation: !ref elevation
pixels: [image, elevation]
tensors:
  image: {shape: [2, height, width], dtype: float32}
  elevation: {shape: [1, height, width], dtype: float32}
""",
    )
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(1)):
        prepared = spec.preprocess({"optical": optical, "dem": dem})
    assert tasks == []
    assert prepared["elevation"].gs.geobox == optical.gs.geobox
    assert dem.gs.geobox == grid
    values = spec.transform(spec.model_inputs(spec.encode(prepared)))
    assert values["image"].shape == (2, 8, 8)
    torch.testing.assert_close(values["elevation"], torch.full((1, 8, 8), 7.0))
