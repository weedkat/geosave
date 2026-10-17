from __future__ import annotations

from datetime import UTC, datetime

import dask.array as da
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import yaml
from odc.geo.geobox import GeoBox
from upath import UPath

from geosave.geodata import (
    GeoDataTree,
    GeoCatalog,
    GeoDataFrame,
    open_vector,
    raster,
    tree,
)
from geosave.model.spec import CallSpec, ModelSpec, RasterRequirement, Ref

from tests.geodata.conftest import item_table


@pytest.mark.parametrize(
    "field",
    ["tiling", "model_inputs", "aggregation", "postprocessing", "exports"],
)
def test_prediction_fields_are_rejected(field):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModelSpec.model_validate({"schema_version": 2, "rasters": {}, field: {}})


def test_python_and_yaml_round_trip_preserves_references_and_literals(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
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
    written = """schema_version: 2
rasters:
  optical:
    variables: [red, nir]
chips:
  size: 128
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
        """schema_version: 2
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
        "schema_version: 2\nrasters: {}\nrasters: {}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, call: builtins.list}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: !ref [model]}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: !unknown model}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {!ref key: 1}}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: {1: value}}",
        "schema_version: 2\nrasters: {1: {}}",
        "schema_version: 2\nrasters: {}\npreprocessing: {1: {call: builtins.dict}}",
        "schema_version: 2\nrasters: {}\npreprocessing:\n  x: {call: builtins.dict, kwargs: &cycle {self: *cycle}}",
    ],
)
def test_invalid_yaml_is_rejected(tmp_path, body):
    path = tmp_path / "model.yaml"
    path.write_text(body)

    with pytest.raises((ValueError, yaml.YAMLError)):
        ModelSpec.load(path)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 1},
        {"schema_version": 3},
        {"extra": {}},
        {"preprocessing": {"bad.name": {"call": "builtins.dict"}}},
        {"preprocessing": {"value": {"call": "builtins.dict", "references": {}}}},
        {
            "preprocessing": {
                "value": {"call": "builtins.dict", "kwargs": {"x": float("nan")}}
            }
        },
        {
            "preprocessing": {
                "value": {"call": "builtins.dict", "kwargs": {"x": object()}}
            }
        },
    ],
)
def test_invalid_python_declarations_fail(change):
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"schema_version": 2, "rasters": {}, **change})


def test_schema_version_is_explicit():
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"rasters": {}})


def test_old_sources_field_is_rejected():
    with pytest.raises(ValueError):
        ModelSpec.model_validate({"schema_version": 2, "sources": {}})


def test_yaml_reference_support_does_not_modify_safe_loader():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load("value: !ref input")


def _spec(**fields: object) -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
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
        """schema_version: 2
rasters: {optical: {channels: 1}}
preprocessing:
  image:
    call: deliberately_uninstalled.module.function
    kwargs: {data: !ref optical}
inputs:
  image: !ref image
  raw: !ref optical
context:
  call: deliberately_uninstalled.module.context
  kwargs: {raster: !ref image}
transforms:
  image:
    - {name: Normalize, init_args: {mean: [0.5], std: [0.25]}}
"""
    )

    spec = ModelSpec.load(path)

    assert spec.inputs["image"] == Ref("image")
    assert spec.context == CallSpec(
        call="deliberately_uninstalled.module.context",
        kwargs={"raster": Ref("image")},
    )
    assert spec.pixel_inputs == ("image", "raw")
    assert spec.input_rasters == ("image", "optical")
    assert spec.transforms["image"][0].init_args == {"mean": [0.5], "std": [0.25]}
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


@pytest.mark.parametrize("name", ["absent", "centre"])
def test_transforms_must_name_a_pixel_input(name):
    with pytest.raises(ValueError, match=f"Transforms name .*{name}"):
        _spec(
            inputs={
                "image": Ref("image"),
            },
            context=CallSpec(call="builtins.dict", kwargs={"centre": Ref("image")}),
            transforms={name: [{"name": "Normalize"}]},
        )


def test_cuts_round_trip(tmp_path):
    spec = _spec(
        frames={
            "windows": {"image": {"slots": ["-4M", "-3M", "-2M", "-1M"]}},
            "stride": 2,
        },
        chips={"size": [224, 256], "overlap": 32, "window": "hann"},
    )

    restored = ModelSpec.load(spec.save(tmp_path / "model_spec.yaml"))

    assert restored == spec
    assert restored.chips is not None and restored.chips.shape == (224, 256)
    assert (
        restored.frames is not None
        and len(restored.frames.windows["image"]["slots"]) == 4
    )
    assert restored.frames.stride == 2


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
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["red"]}},
            "preprocessing": {
                "image": {
                    "call": f"{__name__}.scaled",
                    "kwargs": {"data": Ref("optical")},
                }
            },
            "inputs": {"image": Ref("image")},
            "context": {
                "call": f"{__name__}.centre",
                "kwargs": {"raster": Ref("image")},
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


def test_model_inputs_resolve_prepared_rasters_and_add_their_context():
    image = _optical()[["red"]]

    inputs = _processing().model_inputs({"image": image, "label": xr.Dataset()})

    assert inputs["image"] is image
    assert inputs["centre"] == 1.0


def test_model_inputs_take_cached_context_without_encoding():
    inputs = _processing().model_inputs({"image": _optical()}, context={"centre": 5.0})

    assert inputs["centre"] == 5.0


def test_context_cannot_replace_a_pixel_input():
    with pytest.raises(ValueError, match="image"):
        _processing().model_inputs({"image": _optical()}, context={"image": 0})


def test_a_spec_without_context_encodes_nothing():
    assert _processing(context=None).model_context({"image": _optical()}) == {}


def test_context_must_return_a_dict():
    spec = _processing(
        context={"call": "builtins.list", "kwargs": {}},
    )

    with pytest.raises(TypeError, match="dict"):
        spec.model_context({"image": _optical()})


def test_a_key_reference_round_trips_through_yaml(tmp_path):
    spec = _spec(
        inputs={"image": Ref("optical")},
        context=CallSpec(
            call="builtins.dict",
            kwargs={"azimuth": Ref('optical["view:sun_azimuth"]')},
        ),
    )

    assert ModelSpec.load(spec.save(tmp_path / "model_spec.yaml")) == spec


def test_model_spec_rejects_undeclared_preprocessing_roots() -> None:
    with pytest.raises(ValueError, match="declared rasters.*scale.*value"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {},
                "preprocessing": {
                    "scaled": {
                        "call": Ref("scale"),
                        "kwargs": {"value": Ref("value")},
                    }
                },
            }
        )


def test_context_reads_only_declared_rasters():
    with pytest.raises(ValueError, match="Context references .*row"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {"image": {"channels": 1}},
                "context": {"call": "builtins.dict", "kwargs": {"t": Ref("row.times")}},
            }
        )


def test_context_reads_only_what_the_inputs_read():
    with pytest.raises(ValueError, match="Context references .*optical"):
        ModelSpec.model_validate(
            {
                "schema_version": 2,
                "rasters": {"optical": {"channels": 1}},
                "preprocessing": {
                    "image": {"call": "builtins.dict", "kwargs": {"d": Ref("optical")}}
                },
                "inputs": {"image": Ref("image")},
                "context": {"call": "builtins.dict", "kwargs": {"r": Ref("optical")}},
            }
        )


def test_load_rasters_selects_what_each_recipe_loads(
    serve, red, coordinate_anchor, stac_recipe
):
    serve({"https://primary.test/stac": red})
    model = ModelSpec(
        schema_version=2,
        rasters={"image": RasterRequirement(variables=("red",), stac=stac_recipe)},
    )

    rasters = model.load_rasters(coordinate_anchor)

    assert list(rasters["image"].data_vars) == ["red"]


def test_load_rasters_refuses_rasters_without_a_recipe(coordinate_anchor, stac_recipe):
    model = ModelSpec(
        schema_version=2,
        rasters={
            "image": RasterRequirement(variables=("red",), stac=stac_recipe),
            "dem": RasterRequirement(variables=("height",)),
        },
    )

    with pytest.raises(ValueError, match=r"no `stac` block: \['dem'\]"):
        model.load_rasters(coordinate_anchor)


WHEN = datetime(2025, 6, 1, tzinfo=UTC)


def doubled(*, data: xr.Dataset) -> xr.Dataset:
    return data * 2


def _prepare_spec() -> ModelSpec:
    return ModelSpec.model_validate(
        {
            "schema_version": 2,
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


def _prepare_optical(index: int) -> xr.Dataset:
    left = 500_000 + index * 1000
    grid = GeoBox.from_bbox(
        (left, 9_000_000, left + 80, 9_000_000 + 80), "EPSG:32748", resolution=10
    )
    plane = np.arange(64, dtype="float32").reshape(8, 8) + 100 * index
    return raster({"red": (("y", "x"), plane), "nir": (("y", "x"), plane + 0.5)}, grid)


def _raw(root) -> GeoDataFrame:
    """Write two samples, optical beside a label, and return their item table."""
    tables = []
    for index in range(2):
        optical = _prepare_optical(index)
        label = raster(
            {"label": (("y", "x"), np.full((8, 8), index, "uint8"))},
            optical.gs.geobox,
        )
        sample = tree({"optical": optical, "label": label})
        # One file per band, so each asset is one variable named by its key.
        files = sample.gs.to_cog(f"{root}/s{index}/{{collection}}/{{band}}.tif")
        tables.append(item_table(files, prefix=f"s{index}", datetime=WHEN))
    return open_vector(GeoCatalog.concat(tables).gs.to_geoparquet(root / "raw.parquet"))


def test_prepared_pixels_are_the_preprocessed_raw_pixels(tmp_path):
    spec = _prepare_spec()

    path = spec.prepare(_raw(tmp_path / "raw"), tmp_path / "prepared", keep=["label"])

    prepared = open_vector(path)
    assert sorted(prepared["collection"].unique()) == ["image", "label"]
    assert ModelSpec.load(tmp_path / "prepared" / ModelSpec.filename) == spec
    windows = prepared.gs.sample()
    assert len(windows) == 2
    assert windows["id"].is_unique
    rasters = windows.iloc[1].gs.to_xarray().gs.rasters
    assert list(rasters["image"].data_vars) == ["nir"]
    np.testing.assert_array_equal(
        rasters["image"]["nir"].values,
        2 * _prepare_optical(1)["nir"].values,
    )
    assert (rasters["label"]["label"].values == 1).all()


def test_pixels_stay_lazy_until_they_are_written(tmp_path, monkeypatch):
    raw = _raw(tmp_path / "raw")
    written = []
    to_cog = GeoDataTree.to_cog

    def spy(self, *args, **kwargs):
        written.append(
            all(
                isinstance(r[v].data, da.Array)
                for r in self.rasters.values()
                for v in r
            )
        )
        return to_cog(self, *args, **kwargs)

    monkeypatch.setattr(GeoDataTree, "to_cog", spy)

    _prepare_spec().prepare(raw, tmp_path / "prepared", keep=["label"])

    assert written == [True, True]


def test_an_existing_output_is_refused(tmp_path):
    (tmp_path / "prepared").mkdir()

    with pytest.raises(FileExistsError):
        _prepare_spec().prepare(
            gpd.GeoDataFrame(), tmp_path / "prepared", keep=["label"]
        )


def test_keep_given_as_text_is_refused(tmp_path):
    with pytest.raises(TypeError, match="keep"):
        _prepare_spec().prepare(gpd.GeoDataFrame(), tmp_path / "prepared", keep="label")
