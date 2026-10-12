from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from pydantic import ValidationError

from geosave_engine.geodata.core.constructors import raster
from geosave_engine.model.spec import FieldRequirement, RasterRequirement


def test_persisted_raster_requirements_need_no_acquisition_identity():
    requirement = RasterRequirement(variables=("nir", "red"))
    assert requirement.stac is None


@pytest.mark.parametrize(
    "selector",
    [{}, {"variables": ["red"], "channels": 1}, {"channels": 0}, {"channels": -1}],
)
def test_exactly_one_nonempty_raster_selector_is_required(selector):
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate(selector)


def test_positional_metadata_accepts_only_wildcard_variable_names():
    with pytest.raises(ValidationError, match="Metadata"):
        RasterRequirement.model_validate(
            {"channels": 2, "attrs": {"data_vars": {"red": {}}}}
        )
    RasterRequirement.model_validate({"channels": 2, "attrs": {"data_vars": {"*": {}}}})


def test_an_empty_one_of_fails_at_spec_creation():
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate(
            {
                "variables": ["red"],
                "attrs": {"data_vars": {"red": {"one_of": {"scale_factor": []}}}},
            }
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"variables": []},
        {"variables": ["red", "red"]},
        {"dims": ["y", "y"]},
        {"attrs": {"data_vars": {"missing": {}}}},
    ],
)
def test_ambiguous_or_unreachable_requirements_are_rejected(changes):
    with pytest.raises(ValidationError):
        RasterRequirement.model_validate({"variables": ["red"], **changes})


def test_conflicting_predicates_fail_at_spec_creation():
    with pytest.raises(ValidationError, match="Conflicting"):
        RasterRequirement.model_validate(
            {
                "variables": ["red"],
                "attrs": {
                    "data_vars": {
                        "red": {
                            "equals": {"scale_factor": 1},
                            "one_of": {"scale_factor": [2]},
                        }
                    }
                },
            }
        )


@pytest.mark.parametrize("resolution", [0, -1, (10, 0), (10, -1), float("inf")])
def test_resolution_must_be_positive_and_finite(resolution):
    with pytest.raises(ValueError):
        RasterRequirement(variables=("red",), resolution=resolution)


def _scene(**attrs) -> xr.Dataset:
    grid = GeoBox.from_bbox((0, 0, 40, 40), "EPSG:32633", resolution=10)
    pixels = da.zeros((4, 4), chunks=2, dtype="uint16")
    scene = raster(
        {name: (("y", "x"), pixels) for name in ("red", "nir", "swir")}, grid
    )
    scene["red"].attrs.update(attrs)
    return scene


def test_variables_are_selected_lazily_in_their_declared_order():
    selected = RasterRequirement(variables=("nir", "red")).select_raster(_scene())

    assert list(selected.data_vars) == ["nir", "red"]
    assert isinstance(selected["nir"].data, da.Array)


def test_channels_select_the_first_variables():
    selected = RasterRequirement(channels=2).select_raster(_scene())

    assert list(selected.data_vars) == ["red", "nir"]


@pytest.mark.parametrize(
    ("requirement", "error", "match"),
    [
        ({"variables": ["blue"]}, KeyError, "blue"),
        ({"channels": 4}, ValueError, "4 channels"),
        ({"variables": ["red"], "coordinates": ["time"]}, KeyError, "time"),
        ({"variables": ["red"], "resolution": 20}, ValueError, "resolution"),
        ({"variables": ["red"], "dims": ["x", "y"]}, ValueError, "dims"),
        ({"variables": ["red"], "dtypes": ["float32"]}, ValueError, "dtype"),
    ],
)
def test_an_unmet_requirement_is_refused(requirement, error, match):
    with pytest.raises(error, match=match):
        RasterRequirement.model_validate(requirement).select_raster(_scene())


def test_a_raster_without_a_grid_is_refused_when_a_crs_is_required():
    plain = xr.Dataset({"red": (("y", "x"), np.zeros((2, 2)))})

    with pytest.raises(ValueError, match="CRS"):
        RasterRequirement(variables=("red",), require_crs=True).select_raster(plain)


def test_a_stack_is_not_a_raster():
    with pytest.raises(TypeError, match="Dataset"):
        RasterRequirement(channels=1).select_raster(xr.DataTree())


def test_attrs_compare_after_decoding_gdal_text():
    requirement = RasterRequirement.model_validate(
        {"variables": ["red"], "attrs": {"data_vars": {"*": {"equals": {"offset": 0}}}}}
    )

    selected = requirement.select_raster(_scene(offset="0"))

    assert selected["red"].attrs["offset"] == "0"


def test_a_wildcard_covers_every_selected_variable():
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red", "nir"],
            "attrs": {"data_vars": {"*": {"required": ["offset"]}}},
        }
    )

    with pytest.raises(ValueError, match=r"data_vars\.nir\.offset is required"):
        requirement.select_raster(_scene(offset=0))


@pytest.mark.parametrize(
    ("requirement", "found", "match"),
    [
        ({"required": ["scale"]}, {}, r"root\.scale is required"),
        ({"required": ["scale"]}, {"scale": None}, "required"),
        ({"equals": {"scale": 1}}, {"scale": 2}, "must equal 1"),
        ({"one_of": {"scale": [1, 2]}}, {"scale": "3"}, r"one of \(1, 2\)"),
    ],
)
def test_unmet_attrs_are_refused_by_key(requirement, found, match):
    with pytest.raises(ValueError, match=match):
        FieldRequirement.model_validate(requirement).validate_attrs(found, where="root")


def test_met_attrs_pass():
    requirement = FieldRequirement(equals={"bands": [0, 1]}, one_of={"scale": (1, 2)})

    requirement.validate_attrs({"bands": "[0, 1]", "scale": np.int16(2)}, where="root")
