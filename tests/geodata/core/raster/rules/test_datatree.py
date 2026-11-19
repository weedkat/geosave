"""Restructuring rules reject incompatible nodes without altering raster data."""

import dask
import numpy as np
import pytest
import xarray as xr

from geosave.geodata.core.raster.rules.datatree import (
    compatible_concat,
    compatible_merge,
    splittable,
)
from geosave.geodata.errors import RasterValidationError
from tests.geodata.conftest import _refuse, build_raster


def observations():
    source = build_raster(times=2, packed=True).chunk()
    source = source.assign_coords(time=source.time.isel(time=[0, 0]).values)
    source.red.attrs["units"] = "1"
    source.nir.attrs["units"] = "K"
    return source.gs.to_tree()["ds"].gs.split("time", names=["a", "b"])["time"]


def test_concat_rule_preserves_repeated_labels_and_independent_variable_units():
    tree = observations()
    before = tree.copy(deep=False)

    with dask.config.set(scheduler=_refuse):
        assert tree.gs.validate(compatible_concat("time")) is None
        joined = tree.gs.concat("time").gs.to_dataset()

    xr.testing.assert_identical(tree, before)
    np.testing.assert_array_equal(
        joined.time.values, [tree["a"].time.values, tree["b"].time.values]
    )
    assert joined.red.attrs["units"] == "1"
    assert joined.nir.attrs["units"] == "K"


@pytest.mark.parametrize(
    "field", ["dtype", "units", "flags", "packing", "encoding", "unsigned"]
)
def test_concat_rule_reports_the_incompatible_node_and_variable(field):
    tree = observations()
    other = tree["b"].gs.to_dataset()
    if field == "dtype":
        other["red"] = other.red.astype("float32")
    elif field == "units":
        other.red.attrs["units"] = "m"
    elif field == "flags":
        other.red.attrs.update(flag_values=[1, 2], flag_meanings="land water")
    elif field == "packing":
        other.red.attrs["scale_factor"] = 0.1
    elif field == "encoding":
        other.red.encoding["dtype"] = "uint8"
    else:
        other.red.attrs["_Unsigned"] = "true"
    tree["b"] = xr.DataTree(other)

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError) as error,
    ):
        tree.gs.validate(compatible_concat("time"))

    assert error.value.path == "b"
    assert error.value.variable == "red"
    assert error.value.operation == "concat"


def test_concat_rule_rejects_other_dimension_indexes_before_reading_pixels():
    tree = observations()
    tree["a"] = xr.DataTree(tree["a"].gs.to_dataset().expand_dims(level=[1, 2]))
    tree["b"] = xr.DataTree(tree["b"].gs.to_dataset().expand_dims(level=[2, 3]))

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError) as error,
    ):
        tree.gs.validate(compatible_concat("time"))

    assert error.value.path == "b"
    assert "align" in str(error.value).lower()


@pytest.mark.parametrize("change", ["coordinate", "variables", "grid"])
def test_concat_rule_reports_the_child_that_cannot_be_stacked(change):
    tree = observations()
    other = tree["b"].gs.to_dataset()
    if change == "coordinate":
        other = other.drop_vars("time")
    elif change == "variables":
        other = other.drop_vars("nir")
    else:
        other = other.assign_coords(x=other.x + 100)
    tree["b"] = xr.DataTree(other)

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError) as error,
    ):
        tree.gs.validate(compatible_concat("time"))

    assert error.value.path == "b"
    assert error.value.operation == "concat"


def test_concat_rule_requires_immediate_children():
    tree = xr.DataTree(children={"observations": observations().copy()})

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError, match="immediate raster-leaf"),
    ):
        tree.gs.validate(compatible_concat("time"))


def test_concat_rule_rejects_spatial_concatenation():
    tree = observations()

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError, match="non-spatial"),
    ):
        tree.gs.validate(compatible_concat("x"))


@pytest.mark.parametrize(
    "dim,message",
    [
        ("x", "spatial"),
        ("missing", "one-dimensional coordinate"),
        ("band", "No raster variable"),
    ],
)
def test_split_rule_names_the_unsplittable_dimension(dim, message):
    source = build_raster(times=2).chunk().assign_coords(band=("band", [1]))
    leaf = source.gs.to_tree()["ds"]

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError, match=message) as error,
    ):
        leaf.gs.validate(splittable(dim))

    assert error.value.operation == "split"


def test_split_rule_requires_one_raster_leaf():
    tree = build_raster(times=2).gs.to_tree()

    with pytest.raises(RasterValidationError, match="one raster leaf"):
        tree.gs.validate(splittable("time"))


def test_split_rule_refuses_static_dimension_beside_static_variables():
    source = build_raster(times=2).rename(time="static")
    source["height"] = source.red.isel(static=0, drop=True)

    with pytest.raises(RasterValidationError, match="Rename dimension 'static'"):
        source.gs.to_tree()["ds"].gs.validate(splittable("static"))


def test_merge_rule_names_the_child_repeating_a_variable():
    source = build_raster()
    tree = xr.DataTree.from_dict({"left": source[["red"]], "right": source[["red"]]})

    with (
        dask.config.set(scheduler=_refuse),
        pytest.raises(RasterValidationError, match="Duplicate variables") as error,
    ):
        tree.gs.validate(compatible_merge)

    assert error.value.path == "right"
    assert error.value.operation == "merge"
