from tests.conftest import write_cogs
from datetime import datetime as dt

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr


from geosave.geodata import GeoDataset
from geosave.geodata.core.raster.rules.datatree import same_geobox, same_grid
from tests.geodata.conftest import build_raster, flat


def test_raster_accessor_survives_xarray_selection(raster: xr.Dataset) -> None:
    selected = raster.isel(x=slice(0, 1))

    assert isinstance(selected.gs, GeoDataset)
    assert selected.gs.variables == ("red", "nir")
    geobox = selected.gs.geobox
    assert geobox is not None
    assert geobox.width == 1


def test_times_format_as_a_datetime_index_on_a_raster_and_a_band() -> None:
    raster = build_raster(times=2)

    assert isinstance(raster.gs.times, pd.DatetimeIndex)
    assert raster.gs.times.strftime("%Y%m%d").tolist() == ["20250601", "20250602"]
    assert raster.red.gs.times.equals(raster.gs.times)


@pytest.mark.parametrize(
    ("labels", "span"),
    [
        (["2018-12-26"], ("2018-12-26T00:00:00", "2018-12-26T00:00:00")),
        (
            ["2018-12-27", "2018-12-26"],
            ("2018-12-26T00:00:00", "2018-12-27T00:00:00"),
        ),
    ],
    ids=["an instant", "oldest to newest"],
)
def test_instants_span_from_the_oldest_to_the_newest_label(
    labels: list[str], span: tuple[str, str]
) -> None:
    raster = build_raster(times=len(labels)).assign_coords(
        time=np.array(labels, dtype="datetime64[ns]")
    )

    assert raster.gs.timespan == tuple(dt.fromisoformat(edge) for edge in span)


def _foreign() -> xr.Dataset:
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from odc.geo.xr import xr_coords

    grid = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")
    built = xr.Dataset(
        {"red": (("latitude", "longitude"), np.ones(grid.shape))},
        coords=xr_coords(grid),
    )
    built.red.encoding["grid_mapping"] = "spatial_ref"
    return built


def test_gs_reads_native_spatial_names() -> None:
    foreign = _foreign()
    tree = xr.DataTree.from_dict({"/optical": foreign})

    for held in (foreign, foreign.red):
        held.gs.validate()
        assert held.gs.geobox == foreign.odc.geobox
    tree.gs.validate()
    assert tree.gs.common_geobox() == foreign.odc.geobox


def test_a_band_read_from_disk_uses_native_cf_precision(tmp_path) -> None:
    from geosave.geodata import open_raster

    (written,) = flat(
        write_cogs(build_raster(packed=True), tmp_path, format_paths="scene.tif")
    )

    with open_raster(written) as raw:
        decoded = raw.gs.mask_and_scale()
        expected = xr.decode_cf(raw)
        assert decoded.red.dtype == expected.red.dtype
        assert decoded.red.encoding["dtype"] == raw.red.dtype
        xr.testing.assert_allclose(decoded.red, expected.red)


def _dated(labels, *, dims=("time",), as_dataset=False):
    """Lazy raster whose time labels sit on the given dimension."""
    times = np.array(labels, dtype="datetime64[ns]")
    raster = xr.DataArray(
        da.ones((times.size, 2), chunks=(1, 2)),
        dims=(*dims, "x"),
        coords={"time": (dims, times)},
    )
    return raster.to_dataset(name="image") if as_dataset else raster


@pytest.mark.parametrize("as_dataset", [False, True])
def test_monthly_labels_state_month_cells_without_reading_pixels(as_dataset):
    raster = _dated(["2025-01-01", "2025-02-01"], as_dataset=as_dataset)

    dated = raster.gs.write_time_bounds("MS")

    np.testing.assert_array_equal(
        dated.start_datetime.values,
        np.array(["2025-01-01", "2025-02-01"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(
        dated.end_datetime.values,
        np.array(["2025-02-01", "2025-03-01"], dtype="datetime64[ns]"),
    )
    original = raster.image if as_dataset else raster
    result = dated.image if as_dataset else dated
    assert result.data is original.data
    assert "start_datetime" not in raster.coords


def test_right_labels_state_the_cell_before_them():
    dated = _dated(["2025-01-05", "2025-01-12"]).gs.write_time_bounds(
        "W", label="right"
    )

    np.testing.assert_array_equal(
        dated.start_datetime.values,
        np.array(["2024-12-29", "2025-01-05"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(dated.end_datetime.values, dated.time.values)


def test_scalar_time_states_scalar_cells():
    dated = _dated(["2025-01-01"]).isel(time=0).gs.write_time_bounds("MS")

    assert dated.start_datetime.ndim == 0
    assert dated.end_datetime.values == np.datetime64("2025-02-01", "ns")


def test_labels_off_the_frequency_are_refused():
    with pytest.raises(ValueError, match="not on a MS edge"):
        _dated(["2025-01-10"]).gs.write_time_bounds("MS")


@pytest.mark.parametrize("as_dataset", [False, True])
def test_timespan_follows_selected_cells_and_empty_time(as_dataset):
    from datetime import datetime

    raster = _dated(["2025-01-01", "2025-02-01"], as_dataset=as_dataset)
    dated = raster.gs.write_time_bounds("MS")

    assert dated.isel(time=1).gs.timespan == (
        datetime(2025, 2, 1),
        datetime(2025, 3, 1),
    )
    assert dated.isel(time=1).drop_vars("time").gs.timespan == (
        datetime(2025, 2, 1),
        datetime(2025, 3, 1),
    )
    assert dated.isel(time=slice(0, 0)).gs.timespan is None
    assert raster.isel(time=slice(0, 0)).gs.timespan is None


def test_validation_requires_an_odc_grid_with_a_crs():
    image = xr.DataArray(da.zeros((3, 4)), dims=("y", "x"), name="image")
    with pytest.raises(ValueError, match="grid"):
        image.gs.validate()
    located = image.assign_coords(y=[2.0, 1.0, 0.0], x=[0.0, 1.0, 2.0, 3.0])
    assert not located.gs.is_georeferenced
    with pytest.raises(ValueError, match="CRS"):
        located.gs.validate()
    located.odc.assign_crs("EPSG:4326").gs.validate()


def test_tree_validation_allows_different_extents_unless_requested():
    source = build_raster()
    tree = xr.DataTree.from_dict(
        {"native": source, "crop": source.isel(x=slice(1, None))}
    )

    # A crop keeps the grid: CRS, resolution and pixel alignment.
    tree.gs.validate(same_grid)
    with pytest.raises(ValueError, match="2 GeoBoxes"):
        tree.gs.validate(same_geobox)


def test_validation_errors_explain_correction_and_preserve_tree_context():
    from geosave.geodata import RasterValidationError

    unlocated = xr.Dataset(
        {"red": (("y", "x"), np.ones((2, 3)))},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )
    tree = xr.DataTree.from_dict({"observations/optical": unlocated})
    with pytest.raises(RasterValidationError) as caught:
        tree.gs.validate()
    error = caught.value
    assert error.path == "observations/optical"
    assert "assign_crs" in str(error)
    assert isinstance(error, ValueError)


def test_cells_follow_a_named_label_coordinate():
    raster = _dated(["2025-01-01", "2025-02-01"]).rename(time="acquired")

    dated = raster.gs.write_time_bounds("MS", dim="acquired")

    assert dated.start_datetime.dims == ("acquired",)


@pytest.fixture
def scene() -> xr.Dataset:
    return build_raster(times=3)


def marked(scene: xr.Dataset) -> xr.Dataset:
    out = scene.gs.write_stac_property("platform", "Sentinel-2B")
    return out.gs.write_stac_property("eo:cloud_cover", [0.4, 12.1, 3.0])


def test_write_scalar_and_per_step_properties(scene: xr.Dataset) -> None:
    out = marked(scene)

    assert out["platform"].dims == ()
    assert out["eo:cloud_cover"].dims == ("time",)
    assert out["eo:cloud_cover"].attrs["stac_property"] == "eo:cloud_cover"
    assert out.gs.stac_properties == {
        "platform": "platform",
        "eo:cloud_cover": "eo:cloud_cover",
    }


def test_write_property_on_a_band(scene: xr.Dataset) -> None:
    out = scene.red.gs.write_stac_property("eo:cloud_cover", [1.0, 2.0, 3.0])

    assert out.gs.stac_properties == {"eo:cloud_cover": "eo:cloud_cover"}


def test_write_property_keeps_missing_values(scene: xr.Dataset) -> None:
    out = scene.gs.write_stac_property("platform", ["a", None, "c"])

    assert out["platform"].values.tolist() == ["a", None, "c"]


def test_unmarked_coordinates_are_not_properties(scene: xr.Dataset) -> None:
    out = scene.assign_coords(month=("time", [2, 2, 3]))

    assert "month" not in out.gs.stac_properties


def test_write_along_another_item_dim(scene: xr.Dataset) -> None:
    out = scene.rename(time="scene").gs.write_stac_property(
        "dw:id", ["a", "b", "c"], item_dim="scene"
    )

    assert out["dw:id"].dims == ("scene",)


def test_write_refuses_wrong_length_missing_dim_and_nested_values(
    scene: xr.Dataset,
) -> None:
    with pytest.raises(ValueError, match="length"):
        scene.gs.write_stac_property("a", [1, 2])
    with pytest.raises(KeyError):
        scene.gs.write_stac_property("a", [1, 2, 3], item_dim="scene")
    with pytest.raises(ValueError, match="scalar"):
        scene.gs.write_stac_property("a", [["msi"], "b", "c"])
    with pytest.raises(ValueError, match="scalar"):
        scene.gs.write_stac_property("a", {"k": 1})


@pytest.mark.parametrize("name", ["time", "x", "red"])
def test_write_refuses_names_of_dims_and_variables(
    scene: xr.Dataset, name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        scene.gs.write_stac_property(name, "v")


def test_write_refuses_the_band_name(scene: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="red"):
        scene.red.gs.write_stac_property("red", "v")


@pytest.mark.parametrize(
    "operation",
    [
        lambda ds: ds.isel(time=[0, 2]),
        lambda ds: ds.sel(time=ds.time[1:]),
        lambda ds: ds[["red"]],
        lambda ds: ds * 2,
        lambda ds: ds.where(ds.red > 0),
        lambda ds: ds.mean(["x", "y"]),
        lambda ds: xr.concat([ds, ds], "time"),
        lambda ds: xr.merge([ds, ds]),
    ],
    ids=["isel", "sel", "select", "multiply", "where", "mean_xy", "concat", "merge"],
)
def test_marks_survive_operations_that_keep_the_coordinate(
    scene: xr.Dataset, operation
) -> None:
    out = operation(marked(scene))

    assert set(out.gs.stac_properties) == {"platform", "eo:cloud_cover"}


def test_per_step_marks_go_with_time_reductions(scene: xr.Dataset) -> None:
    out = marked(scene).mean("time")

    assert "eo:cloud_cover" not in out.coords
    assert out["platform"].attrs["stac_property"] == "platform"


@pytest.mark.parametrize("fmt", ["netcdf", "zarr"])
def test_property_coordinates_round_trip_through_files(
    scene: xr.Dataset, tmp_path, fmt: str
) -> None:
    raster = marked(scene)
    if fmt == "netcdf":
        path = tmp_path / "scene.nc"
        raster.to_netcdf(path)
        back = xr.open_dataset(path)
    else:
        path = tmp_path / "scene.zarr"
        raster.to_zarr(path)
        back = xr.open_zarr(path)

    assert back.gs.stac_properties == raster.gs.stac_properties
    assert back["eo:cloud_cover"].values.tolist() == [0.4, 12.1, 3.0]
