from __future__ import annotations

import pytest
import xarray as xr
from pydantic import ValidationError

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


def test_raster_selection_is_not_settled():
    with pytest.raises(NotImplementedError, match="select_raster"):
        RasterRequirement(variables=("red",)).select_raster(xr.Dataset())


def test_attrs_validation_is_not_settled():
    requirement = FieldRequirement(required=("scale_factor",))

    with pytest.raises(NotImplementedError, match="validate_attrs"):
        requirement.validate_attrs({}, where="root")
