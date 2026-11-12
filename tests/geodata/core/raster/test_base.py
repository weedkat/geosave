from tests.conftest import write_cogs
from datetime import datetime as dt

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dask.callbacks import Callback

from geosave.geodata import GeoDataset
from geosave.geodata.core.raster.chips import Chips
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
        (["2018-12-26T10:30:31"], ("2018-12-26T10:30:31", "2018-12-26T10:30:31")),
        (
            ["2018-12-27", "2018-12-26"],
            ("2018-12-26T00:00:00", "2018-12-27T00:00:00"),
        ),
    ],
    ids=["an instant", "an instant with seconds", "oldest to newest"],
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
    assert tree.gs.geobox == foreign.odc.geobox


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


@pytest.mark.parametrize(
    "freq, origin, labels, starts",
    [
        ("ME", "start_day", ["2025-01-31", "2025-02-28"], ["2024-12-31", "2025-01-31"]),
        ("7D", "end", ["2025-01-05", "2025-01-12"], ["2024-12-29", "2025-01-05"]),
    ],
)
def test_time_bounds_resolve_native_label_defaults(freq, origin, labels, starts):
    raster = _dated(labels).rename(time="acquired")

    dated = raster.gs.write_time_bounds(freq, dim="acquired", origin=origin)

    np.testing.assert_array_equal(
        dated.start_datetime.values, np.array(starts, dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(dated.end_datetime.values, dated.acquired.values)
    assert dated.data is raster.data


def test_cells_follow_an_auxiliary_time_dimension():
    raster = _dated(["2025-01-01", "2025-02-01"], dims=("observation",))

    dated = raster.gs.write_time_bounds("MS")

    assert dated.start_datetime.dims == ("observation",)
    assert dated.end_datetime.dims == ("observation",)
    assert dated.sizes == raster.sizes


def test_scalar_time_states_scalar_cells():
    dated = _dated(["2025-01-01"]).isel(time=0).gs.write_time_bounds("MS")

    assert dated.start_datetime.ndim == 0
    assert dated.end_datetime.values == np.datetime64("2025-02-01", "ns")


def test_labels_off_the_frequency_are_refused():
    with pytest.raises(ValueError, match="not on a MS edge"):
        _dated(["2025-01-10"]).gs.write_time_bounds("MS")


def test_cells_need_a_time_coordinate():
    with pytest.raises(KeyError, match="temporal coordinate"):
        xr.DataArray([1.0], dims="x").gs.write_time_bounds("MS")


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


def test_tree_validation_allows_different_grids_unless_requested():
    source = build_raster()
    tree = xr.DataTree.from_dict(
        {"native": source, "crop": source.isel(x=slice(1, None))}
    )
    tree.gs.validate()
    with pytest.raises(ValueError, match="2 grids"):
        tree.gs.validate(same_grid=True)


def test_native_axes_support_inspection_validation_and_chips():
    source = build_raster().rename(y="latitude", x="longitude")
    source.gs.validate()
    assert source.red.gs.geobox == source.odc.geobox
    chips = Chips([source], size=3, constant_value=0)
    predictions = []
    for key in chips:
        chip = chips[key]
        assert chip.gs.spatial_dims == ("latitude", "longitude")
        predictions.append((key, np.ones(chip.gs.geobox.shape)))
    pixels = chips.merge(predictions)[0]
    reference = chips.xarray[0]
    merged = xr.DataArray(
        pixels, dims=reference.gs.spatial_dims, coords=reference.coords
    )
    merged.encoding["grid_mapping"] = str(reference.odc.crs_coord.name)
    assert merged.dims == ("latitude", "longitude")
    assert merged.gs.geobox == source.gs.geobox


def test_a_gcp_grid_is_not_implemented():
    from odc.geo.gcp import GCPGeoBox, GCPMapping
    from odc.geo.geom import multipoint
    from odc.geo.xr import xr_coords

    pixels = multipoint([(0, 0), (4, 0), (0, 4), (4, 4)], None)
    world = multipoint([(10, 20), (14, 20), (10, 16), (14, 16)], "EPSG:4326")
    grid = GCPGeoBox((4, 4), GCPMapping(pixels, world))
    image = xr.DataArray(np.ones((4, 4)), dims=("y", "x")).assign_coords(
        xr_coords(grid, dims=("y", "x"))
    )
    for held in (image, image.to_dataset(name="image")):
        with pytest.raises(NotImplementedError, match="ground control points"):
            held.gs.validate()


def test_cf_axes_can_be_inspected_without_renaming():
    source = build_raster().rename(y="north", x="east")
    source.north.attrs["axis"] = "Y"
    source.east.attrs["axis"] = "X"
    source.gs.validate()
    assert source.red.dims == ("north", "east")
    assert source.gs.geobox == build_raster().gs.geobox


def test_empty_and_scalar_inputs_are_inspectable_but_fail_spatial_validation():
    assert xr.Dataset().gs.geobox is None
    scalar = xr.DataArray(1, name="version")
    assert scalar.gs.geobox is None
    for value in (
        xr.Dataset(),
        scalar,
        xr.DataArray(np.zeros((0, 2)), dims=("y", "x")),
    ):
        with pytest.raises(ValueError):
            value.gs.validate()


def test_native_axes_keep_numpy_layout():
    source = build_raster().rename(y="latitude", x="longitude")
    np.testing.assert_array_equal(source.gs.to_numpy(), build_raster().gs.to_numpy())
    np.testing.assert_array_equal(source.red.gs.to_numpy(), source.red.values)


def test_spatial_dims_describes_native_axes_and_tree_agreement_without_pixels():
    source = build_raster().chunk().rename(y="latitude", x="longitude")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        assert source.gs.spatial_dims == ("latitude", "longitude")
        assert source.red.gs.spatial_dims == ("latitude", "longitude")
        tree = xr.DataTree.from_dict(
            {
                "metadata": xr.Dataset(attrs={"title": "scene"}),
                "image": source,
                "crop": source.isel(longitude=slice(1, None)),
            }
        )
        assert tree.gs.spatial_dims == ("latitude", "longitude")
        mixed = xr.DataTree.from_dict({"image": source, "native": build_raster()})
        assert mixed.gs.spatial_dims is None
        assert xr.DataTree().gs.spatial_dims is None
        assert xr.DataArray(1).gs.spatial_dims is None
    assert tasks == []


def test_spatial_dims_reads_cf_axes_without_assuming_dimension_order():
    source = build_raster().rename(y="north", x="east").expand_dims(level=[1], axis=-1)
    source.north.attrs["axis"] = "Y"
    source.east.attrs["standard_name"] = "projection_x_coordinate"
    assert source.gs.spatial_dims == ("north", "east")
    assert source.red.gs.spatial_dims == ("north", "east")


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
