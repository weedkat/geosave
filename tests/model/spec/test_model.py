from __future__ import annotations


import dask.array as da
import numpy as np
import pytest
import xarray as xr
import yaml
from odc.geo.geobox import GeoBox
from upath import UPath

from geosave.geodata import GeoCatalog, GeoDataTree, raster, stac
from geosave.model.spec import CallSpec, ModelSpec, RasterRequirement, Ref


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


def doubled(*, data: xr.Dataset) -> xr.Dataset:
    return data * 2


def _ingest_spec() -> ModelSpec:
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


GRID = GeoBox.from_bbox(
    (500_000, 9_000_000, 500_080, 9_000_080), "EPSG:32748", resolution=10
)
NIR = np.arange(64, dtype="uint16").reshape(8, 8)


def _scene() -> xr.Dataset:
    optical = raster({"nir": (("y", "x"), da.from_array(NIR))}, GRID)
    return optical.expand_dims(time=[np.datetime64("2025-06-01T10:30:00")])


def _label() -> xr.Dataset:
    plane = np.full((8, 8), 3, "uint8")
    label = raster({"label": (("y", "x"), plane)}, GRID)
    return label.expand_dims(time=[np.datetime64("2025-06-01")])


@pytest.fixture
def loaded(monkeypatch):
    """Serve `_scene` as every recipe's load and record the anchors asked for."""
    anchors = []

    def load(spec, anchor):
        anchors.append(anchor)
        return {"optical": _scene()}

    monkeypatch.setattr(ModelSpec, "load_rasters", load)
    return anchors


def test_inference_reads_an_ingested_sample_through_preprocess(tmp_path, loaded):
    spec = _ingest_spec()
    anchor = _label().gs.anchor

    items = spec.ingest(anchor, f"{tmp_path}/{{collection}}/{{band}}.tif")

    assert loaded == [anchor]
    rasters = items.gs.sample().iloc[0].gs.to_xarray().gs.rasters
    assert list(rasters) == ["optical"]
    image = spec.preprocess(rasters)["image"]
    np.testing.assert_array_equal(image["nir"].values.squeeze(), 2 * NIR)


def test_training_samples_pair_the_ingested_table_with_a_label_table(tmp_path, loaded):
    label = _label()
    images = _ingest_spec().ingest(
        label.gs.anchor, f"{tmp_path}/{{collection}}/{{band}}.tif"
    )
    labels = stac.create_catalog(
        stac.create_items(
            label.gs.to_cog(f"{tmp_path}/label/label.tif"), collection="label"
        )
    )

    samples = GeoCatalog.concat([images, labels]).gs.sample()

    assert len(samples) == 1
    rasters = samples.iloc[0].gs.to_xarray().gs.rasters
    assert sorted(rasters) == ["label", "optical"]
    assert (rasters["label"]["label"].values == 3).all()


def test_ingested_pixels_stay_lazy_until_they_are_written(
    tmp_path, loaded, monkeypatch
):
    written = []
    to_cog = GeoDataTree.to_cog

    def spy(self, *args, **kwargs):
        for layer in self.rasters.values():
            for name in layer.data_vars:
                written.append(isinstance(layer[name].data, da.Array))
        return to_cog(self, *args, **kwargs)

    monkeypatch.setattr(GeoDataTree, "to_cog", spy)

    _ingest_spec().ingest(_label().gs.anchor, f"{tmp_path}/{{collection}}.tif")

    assert written == [True]


def test_item_ids_follow_the_given_format(tmp_path, loaded):
    items = _ingest_spec().ingest(
        _label().gs.anchor,
        f"{tmp_path}/{{collection}}.tif",
        id_format="tile_{start:%Y%m%d}",
    )

    assert items["id"].tolist() == ["tile_20250601"]
