from __future__ import annotations

from datetime import UTC, datetime

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from odc.geo.geobox import GeoBox
from upath import UPath

from geosave.geodata import (
    GeoCatalog,
    GeoDataFrame,
    open_vector,
)
from tests.conftest import build_dataset, build_tree
from geosave.geodata.stac.catalog import from_items
from geosave.geodata.io.formats import geotiff
from geosave.model.spec import CallSpec, ModelSpec, RasterRequirement, Ref
from tests.geodata.conftest import item_table, write_collection_cogs
from tests.geodata.conftest import create_file_items

WHEN = datetime(2025, 6, 1, tzinfo=UTC)


def test_python_and_yaml_round_trip_preserves_references_and_literals(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 3,
            "rasters": {"optical": {"channels": 1}},
            "preprocessing": {
                "selected": CallSpec(
                    call=Ref("optical.__getitem__"),
                    kwargs={"key": ["red", "nir"]},
                ),
                "result": CallSpec(
                    call="missing_package.for_this_stage",
                    kwargs={
                        "items": [Ref("selected"), {"ref": "literal"}],
                        "label": "!ref selected",
                    },
                ),
            },
        }
    )
    assert ModelSpec.model_validate(spec.model_dump()) == spec

    path = spec.save(tmp_path / "model_spec.yaml")

    assert "!ref" in path.read_text()
    assert ModelSpec.load(path) == spec


def test_a_saved_spec_states_only_what_was_declared(tmp_path):
    written = """schema_version: 3
rasters:
  optical:
    variables: [red, nir]
inputs:
  image: !ref optical
"""
    source = tmp_path / "written.yaml"
    source.write_text(written)
    spec = ModelSpec.load(source)

    saved = spec.save(tmp_path / "saved.yaml")

    assert yaml.safe_load(saved.read_text().replace("!ref", "")) == yaml.safe_load(
        written.replace("!ref", "")
    )
    assert ModelSpec.load(saved) == spec


def test_parsing_never_imports_or_executes_calls(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 3
rasters: {source: {channels: 1}}
preprocessing:
  result:
    call: deliberately_uninstalled.module.function
    kwargs: {input: !ref source}
"""
    )

    result = ModelSpec.load(path)

    assert result.preprocessing["result"].kwargs["input"] == Ref("source")


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 3\nrasters: {}\nrasters: {}",
        "schema_version: 3\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, call: builtins.list}",
        "schema_version: 3\nrasters: {}\npreprocessing:\n  x: {call: !ref [model]}",
        "schema_version: 3\nrasters: {}\npreprocessing:\n  x: {call: !unknown model}",
        "schema_version: 3\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {!ref key: 1}}",
        "schema_version: 3\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {1: value}}",
        "schema_version: 3\nrasters: {1: {}}",
        "schema_version: 3\nrasters: {}\npreprocessing: {1: {call: builtins.dict}}",
        "schema_version: 3\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: &cycle {self: *cycle}}",
    ],
)
def test_invalid_yaml_is_rejected(tmp_path, body):
    path = tmp_path / "model.yaml"
    path.write_text(body)

    with pytest.raises((ValueError, yaml.YAMLError)):
        ModelSpec.load(path)


def test_yaml_reference_support_does_not_modify_safe_loader():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("value: !ref input")


def _spec(**fields: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 3,
            "rasters": {"optical": {"channels": 1}},
            "preprocessing": {
                "image": {"call": "builtins.dict", "kwargs": {"data": Ref("optical")}}
            },
            **fields,
        }
    )


def test_inputs_round_trip_references_and_calls(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        """schema_version: 3
rasters: {optical: {channels: 1}}
preprocessing:
  image:
    call: deliberately_uninstalled.module.function
    kwargs: {data: !ref optical}
encoding:
  context:
    call: deliberately_uninstalled.module.context
    kwargs: {raster: !ref optical}
inputs:
  image: !ref image
  centre: !ref 'context["centre"]'
pixels: [image]
transforms:
  image:
    call: deliberately_uninstalled.module.normalize
    kwargs: {value: !ref image, mean: [0.5], std: [0.25]}
"""
    )
    spec = ModelSpec.load(path)
    assert spec.inputs["image"] == Ref("image")
    assert ModelSpec.load(spec.save(tmp_path / "saved.yaml")) == spec


@pytest.mark.parametrize(
    "inputs",
    [
        {"image": Ref("missing")},
        {"centre": Ref("missing")},
    ],
)
def test_inputs_must_reference_a_step_or_a_raster(inputs):
    with pytest.raises(ValueError, match=r"Model inputs reference .*missing"):
        _spec(inputs=inputs)


def test_transforms_reject_unavailable_arguments_before_execution():
    with pytest.raises(ValueError, match="Missing stage inputs.*absent"):
        _spec(
            inputs={"image": Ref("image")},
            transforms={
                "image": {"call": "builtins.dict", "kwargs": {"value": Ref("absent")}}
            },
        )


def test_a_spec_round_trips_through_a_remote_upath(bucket):
    spec = _spec(inputs={"image": Ref("image")})
    target = UPath(bucket) / "model_spec.yaml"

    saved = spec.save(target)

    assert saved == target
    assert ModelSpec.load(target) == spec


def scaled(*, data: xr.Dataset) -> xr.Dataset:
    return data * 2


def centre(*, raster: xr.Dataset) -> dict[str, float]:
    return {"centre": float(raster["red"].mean())}


def _processing(**fields: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 3,
            "rasters": {"optical": {"variables": ["red"]}},
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.scaled",
                    "kwargs": {"data": Ref("optical")},
                }
            },
            "inputs": {"image": Ref("image"), "centre": Ref('context["centre"]')},
            "encoding": {
                "context": {
                    "call": f"{__name__}.centre",
                    "kwargs": {"raster": Ref("image")},
                }
            },
            **fields,
        }
    )


def _optical() -> xr.Dataset:
    return xr.Dataset(
        {
            "red": (("y", "x"), da.ones((2, 2), chunks=1)),
            "nir": (("y", "x"), da.zeros((2, 2))),
        }
    )


def test_preprocess_returns_selected_rasters_and_results_lazily():
    results = _processing().preprocess({"optical": _optical(), "label": xr.Dataset()})

    assert sorted(results) == ["image", "optical"]
    assert list(results["optical"].data_vars) == ["red"]
    assert isinstance(results["image"]["red"].data, da.Array)
    assert float(results["image"]["red"].max()) == 2


def test_preprocess_refuses_a_missing_declared_raster():
    with pytest.raises(KeyError, match="optical"):
        _processing().preprocess({"label": xr.Dataset()})


def test_encoding_binds_context_from_the_same_sample_and_keeps_state():
    spec = _processing()
    prepared = spec.preprocess({"optical": _optical()})
    encoded = spec.encode(prepared)
    inputs = spec.model_inputs(encoded)
    assert inputs["image"] is prepared["image"]
    assert inputs["centre"] == 2.0
    assert encoded["optical"] is prepared["optical"]
    assert set(inputs) == {"image", "centre"}


def test_auxiliary_raster_metadata_can_supply_context():
    spec = _processing(
        encoding={
            "context": {
                "call": "builtins.dict",
                "kwargs": {"centre": Ref('optical.attrs["centre"]')},
            }
        }
    )
    prepared = spec.preprocess({"optical": _optical().assign_attrs(centre=7)})
    assert spec.model_inputs(spec.encode(prepared))["centre"] == 7


def test_model_spec_rejects_undeclared_preprocessing_roots():
    with pytest.raises(ValueError, match="Missing stage inputs.*scale.*value"):
        ModelSpec.model_validate(
            {
                "schema_version": 3,
                "rasters": {},
                "preprocessing": {
                    "scaled": {"call": Ref("scale"), "kwargs": {"value": Ref("value")}}
                },
            }
        )


def test_load_rasters_selects_what_each_recipe_loads(
    serve, red, coordinate_anchor, coordinate_grid, stac_recipe
):
    serve({"https://primary.test/stac": red})
    model = ModelSpec(
        schema_version=3,
        rasters={"image": RasterRequirement(variables=("red",), stac=stac_recipe)},
    )

    rasters = model.load_rasters(coordinate_anchor, geobox=coordinate_grid)

    assert list(rasters["image"].data_vars) == ["red"]


def test_load_rasters_refuses_rasters_without_a_recipe(
    coordinate_anchor, coordinate_grid, stac_recipe
):
    model = ModelSpec(
        schema_version=3,
        rasters={
            "image": RasterRequirement(variables=("red",), stac=stac_recipe),
            "dem": RasterRequirement(variables=("height",)),
        },
    )

    with pytest.raises(ValueError, match=r"no `stac` block: \['dem'\]"):
        model.load_rasters(coordinate_anchor, geobox=coordinate_grid)


def doubled(*, data: xr.Dataset) -> xr.Dataset:
    return data * 2


def _ingest_spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 3,
            "rasters": {"optical": {"variables": ["nir"]}},
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.doubled",
                    "kwargs": {"data": Ref("optical")},
                }
            },
            "inputs": {"image": Ref("image")},
        }
    )


GRID = GeoBox.from_bbox(
    (500_000, 9_000_000, 500_080, 9_000_080), "EPSG:32748", resolution=10
)
NIR = np.arange(64, dtype="uint16").reshape(8, 8)


def _scene() -> xr.Dataset:
    optical = build_dataset({"nir": (("y", "x"), da.from_array(NIR))}, GRID)
    return optical.expand_dims(time=[np.datetime64("2025-06-01T10:30:00")])


def _label() -> xr.Dataset:
    plane = np.full((8, 8), 3, "uint8")
    label = build_dataset({"label": (("y", "x"), plane)}, GRID)
    return label.expand_dims(time=[np.datetime64("2025-06-01")])


@pytest.fixture
def loaded(monkeypatch):
    """Serve `_scene` as every recipe's load and record the anchors asked for."""
    anchors = []

    def load(spec, anchor, *, geobox):
        anchors.append(anchor)
        return {"optical": _scene()}

    monkeypatch.setattr(ModelSpec, "load_rasters", load)
    return anchors


def test_inference_reads_an_ingested_sample_through_preprocess(tmp_path, loaded):
    spec = _ingest_spec()
    anchor = _label().gs.anchor

    items = spec.ingest(
        anchor, tmp_path, geobox=GRID, path_format="{collection}/{band}.tif"
    )

    assert loaded == [anchor]
    tree = items.gs.sample()[0].gs.to_datatree()
    rasters = {"optical": tree["optical"].gs.to_dataset()}
    assert list(rasters) == ["optical"]
    image = spec.preprocess(rasters)["image"]
    np.testing.assert_array_equal(image["nir"].values.squeeze(), 2 * NIR)


def test_training_samples_pair_the_ingested_table_with_a_label_table(tmp_path, loaded):
    label = _label()
    images = _ingest_spec().ingest(
        label.gs.anchor, tmp_path, geobox=GRID, path_format="{collection}/{band}.tif"
    )
    labels = from_items(
        create_file_items(
            label.gs.to_cog(tmp_path / "label", path_format="label.tif"),
            collection="label",
        )
    )

    samples = GeoCatalog.concat([images, labels]).gs.sample()

    assert len(samples) == 1
    tree = samples[0].gs.to_datatree()
    rasters = {name: tree[name].gs.to_dataset() for name in ("label", "optical")}
    assert sorted(rasters) == ["label", "optical"]
    assert (rasters["label"]["label"].values == 3).all()


def test_ingested_pixels_stay_lazy_until_they_are_written(
    tmp_path, loaded, monkeypatch
):
    written = []
    write_cog = geotiff.write_cog

    def spy(raster, *args, **kwargs):
        for name in raster.data_vars:
            written.append(isinstance(raster[name].data, da.Array))
        return write_cog(raster, *args, **kwargs)

    monkeypatch.setattr(geotiff, "write_cog", spy)

    _ingest_spec().ingest(
        _label().gs.anchor, tmp_path, geobox=GRID, path_format="{collection}.tif"
    )

    assert written == [True]


def test_item_ids_follow_the_given_format(tmp_path, loaded):
    items = _ingest_spec().ingest(
        _label().gs.anchor,
        tmp_path,
        geobox=GRID,
        path_format="{collection}.tif",
        id_format="tile_{start:%Y%m%d}",
    )

    assert items["id"].tolist() == ["tile_20250601"]


def _prepare_optical(index: int) -> xr.Dataset:
    left = 500_000 + index * 1000
    grid = GeoBox.from_bbox(
        (left, 9_000_000, left + 80, 9_000_000 + 80), "EPSG:32748", resolution=10
    )
    plane = np.arange(64, dtype="float32").reshape(8, 8) + 100 * index
    return build_dataset(
        {"red": (("y", "x"), plane), "nir": (("y", "x"), plane + 0.5)}, grid
    )


def _raw(root) -> GeoDataFrame:
    """Write two samples, optical beside a label, and return their item table."""
    tables = []
    for index in range(2):
        optical = _prepare_optical(index)
        label = build_dataset(
            {"label": (("y", "x"), np.full((8, 8), index, "uint8"))},
            optical.gs.geobox,
        )
        sample = build_tree({"optical": optical, "label": label})
        # One file per band, so each asset is one variable named by its key.
        templates = {name: f"s{index}/{name}/{{band}}.tif" for name in sample.gs.groups}
        files = write_collection_cogs(sample, root, templates)
        tables.append(item_table(files, prefix=f"s{index}", datetime=WHEN))
    return open_vector(GeoCatalog.concat(tables).gs.to_geoparquet(root / "raw.parquet"))


def test_prepared_pixels_are_the_preprocessed_raw_pixels(tmp_path):
    spec = _ingest_spec()

    prepared = spec.prepare(
        _raw(tmp_path / "raw"), tmp_path / "prepared", keep=["label"]
    )

    assert sorted(prepared["collection"].unique()) == ["image", "label"]
    assert ModelSpec.load(tmp_path / "prepared" / ModelSpec.filename) == spec
    restored = open_vector(prepared.gs.to_geoparquet(tmp_path / "prepared.parquet"))
    samples = restored.gs.sample()
    assert len(samples) == 2
    identities = pd.concat(list(samples))[["collection", "id"]]
    assert not identities.duplicated().any()
    folders = []
    for entry in (tmp_path / "prepared").iterdir():
        if entry.is_dir():
            folders.append(entry.name)
    assert sorted(folders) == ["0", "1"]
    tree = samples[1].gs.to_datatree()
    rasters = {name: tree[name].gs.to_dataset() for name in ("image", "label")}
    assert list(rasters["image"].data_vars) == ["nir"]
    np.testing.assert_array_equal(
        rasters["image"]["nir"].values, 2 * _prepare_optical(1)["nir"].values
    )
    assert (rasters["label"]["label"].values == 1).all()


def test_prepare_without_a_path_writes_a_temporary_catalog(tmp_path):
    prepared = _ingest_spec().prepare(_raw(tmp_path / "raw"))

    tree = prepared.gs.sample()[0].gs.to_datatree()
    rasters = {"image": tree["image"].gs.to_dataset()}
    assert list(rasters) == ["image"]
    np.testing.assert_array_equal(
        rasters["image"]["nir"].values, 2 * _prepare_optical(0)["nir"].values
    )


def test_prepare_keeps_pixels_lazy_until_written(tmp_path, monkeypatch):
    raw = _raw(tmp_path / "raw")
    written = []
    write_cog = geotiff.write_cog

    def spy(raster, *args, **kwargs):
        written.append(
            all(isinstance(raster[name].data, da.Array) for name in raster.data_vars)
        )
        return write_cog(raster, *args, **kwargs)

    monkeypatch.setattr(geotiff, "write_cog", spy)

    _ingest_spec().prepare(raw, tmp_path / "prepared", keep=["label"])

    # Two samples, each writing its image and its kept label.
    assert written == [True] * 4


def test_prepare_refuses_an_existing_folder(tmp_path):
    (tmp_path / "prepared").mkdir()

    with pytest.raises(FileExistsError):
        _ingest_spec().prepare(gpd.GeoDataFrame(), tmp_path / "prepared")


def test_prepare_names_a_missing_kept_collection(tmp_path):
    with pytest.raises(KeyError, match="dem"):
        _ingest_spec().prepare(_raw(tmp_path / "raw"), keep=["dem"])
