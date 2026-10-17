from datetime import datetime as dt
from typing import TYPE_CHECKING, assert_type

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geosave.geodata as gd
from geosave.geodata import GeoDataArray, GeoDataset, GeoDataTree
from geosave.geodata.core.raster.create import raster
from tests.geodata.conftest import build_raster, geobox
from tests.geodata.conftest import flat

if TYPE_CHECKING:

    def _assert_accessor_types(
        array: gd.DataArray,
        dataset: gd.Dataset,
        tree: gd.DataTree,
    ) -> None:
        assert_type(array.gs, GeoDataArray)
        assert_type(dataset.gs, GeoDataset)
        assert_type(tree.gs, GeoDataTree)
        assert_type(dataset.isel(x=0).gs, GeoDataset)


def test_raster_accessor_survives_xarray_selection(raster: xr.Dataset) -> None:
    selected = raster.isel(x=slice(0, 1))

    assert isinstance(selected.gs, GeoDataset)
    assert selected.gs.variables == ("red", "nir")
    geobox = selected.gs.geobox
    assert geobox is not None
    assert geobox.width == 1


def test_tree_accessor_reads_settled_structure(tree: xr.DataTree) -> None:
    assert isinstance(tree.gs, GeoDataTree)
    assert tree.gs.groups == ("optical", "infrared")


def test_every_xarray_accessor_names_its_variables(
    raster: xr.Dataset, tree: xr.DataTree
) -> None:
    assert raster.red.gs.variables == ("red",)
    assert raster.gs.variables == ("red", "nir")
    assert tree.gs.variables == ("optical/red", "infrared/nir")


def test_unbucketed_time_covers_only_its_own_instants() -> None:
    raster = build_raster(times=2).assign_coords(
        time=np.array(
            ["2025-06-01T10:23:11", "2025-06-11T10:24:02"], dtype="datetime64[ns]"
        )
    )

    start, end = raster.gs.timespan

    assert (start.hour, start.minute, start.second) == (10, 23, 11)
    assert (end.hour, end.minute, end.second) == (10, 24, 2)


def test_timeless_raster_has_no_timespan(raster: xr.Dataset) -> None:
    assert raster.gs.timespan is None


def test_times_format_as_a_datetime_index_on_a_raster_and_a_band() -> None:
    raster = build_raster(times=2)

    assert isinstance(raster.gs.times, pd.DatetimeIndex)
    assert raster.gs.times.strftime("%Y%m%d").tolist() == ["20250601", "20250602"]
    assert raster.red.gs.times.equals(raster.gs.times)


def test_timeless_data_has_no_times(raster: xr.Dataset) -> None:
    assert raster.gs.times is None
    assert raster.red.gs.times is None


def test_times_leave_chunked_pixels_uncomputed() -> None:
    raster = build_raster(times=2).chunk()

    assert len(raster.gs.times) == 2
    assert isinstance(raster.red.data, da.Array)


def test_geodata_types_are_xarray_at_runtime() -> None:
    assert gd.DataArray is xr.DataArray
    assert gd.Dataset is xr.Dataset
    assert gd.DataTree is xr.DataTree


@pytest.mark.parametrize(
    ("labels", "span"),
    [
        (["2018-12-26"], ("2018-12-26T00:00:00", "2018-12-26T23:59:59.999999")),
        (["2018-12-01"], ("2018-12-01T00:00:00", "2018-12-01T23:59:59.999999")),
        (["2018-01-01"], ("2018-01-01T00:00:00", "2018-01-01T23:59:59.999999")),
        (
            ["2018-12-26", "2018-12-27"],
            ("2018-12-26T00:00:00", "2018-12-27T23:59:59.999999"),
        ),
        (
            ["2018-12-26T10:00:00"],
            ("2018-12-26T10:00:00", "2018-12-26T10:00:59.999999"),
        ),
    ],
    ids=["a date", "a month start", "a year start", "two dates", "an hour"],
)
def test_unbucketed_time_covers_what_its_labels_spell(
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


def test_gs_refuses_a_grid_not_named_y_x() -> None:
    foreign = _foreign()
    tree = xr.DataTree.from_dict({"/optical": foreign})

    for held in (foreign, foreign.red, tree):
        with pytest.raises(ValueError, match="rename it first"):
            _ = held.gs


def test_gs_reads_the_grid_once_it_is_renamed() -> None:
    foreign = _foreign()

    renamed = foreign.rename({"latitude": "y", "longitude": "x"})

    assert renamed.gs.geobox == foreign.odc.geobox
    assert renamed.gs.write_crs().y.attrs["standard_name"] == "latitude"


@pytest.mark.parametrize("kind", ["raster", "array", "tree"])
@pytest.mark.parametrize("lazy", [False, True])
def test_mask_and_scale_decodes_values_and_keeps_the_source(
    kind: str, lazy: bool
) -> None:
    scene = build_raster(packed=True)
    scene.attrs["title"] = "Stored scene"
    scene.red.attrs.update(long_name="Red", units="1", add_offset=np.float32(0.5))
    if lazy:
        scene = scene.chunk({"y": 1, "x": 1})
    if kind == "array":
        source = scene.red
    elif kind == "tree":
        source = xr.DataTree.from_dict(
            {"/": xr.Dataset(attrs={"title": "Stack"}), "/optical": scene}
        )
    else:
        source = scene
    before = source.copy(deep=True)

    decoded = source.gs.mask_and_scale()
    band = (
        decoded["optical/red"]
        if kind == "tree"
        else decoded
        if kind == "array"
        else decoded.red
    )

    assert isinstance(decoded, type(source))
    assert isinstance(band.data, da.Array) == lazy
    np.testing.assert_allclose(band.compute().values, [[0.6, 0.7], [0.8, np.nan]])
    assert "scale_factor" not in band.attrs
    assert "add_offset" not in band.attrs
    assert "_FillValue" not in band.attrs
    assert band.attrs["long_name"] == "Red"
    assert band.attrs["units"] == "1"
    assert band.gs.geobox == scene.gs.geobox
    if kind != "array":
        assert decoded.attrs == source.attrs
    xr.testing.assert_identical(decoded.gs.mask_and_scale(), decoded)
    xr.testing.assert_identical(source, before)


def test_times_reads_a_scalar_time_coordinate() -> None:
    from tests.geodata.conftest import build_raster

    one_date = build_raster(times=2).isel(time=0)

    assert one_date.gs.times.strftime("%Y%m%d").tolist() == [
        build_raster(times=2).gs.times[0].strftime("%Y%m%d")
    ]


def test_a_band_read_from_disk_decodes_to_float32(tmp_path) -> None:
    from geosave.geodata import open_raster

    (written,) = flat(build_raster(packed=True).gs.to_cog(f"{tmp_path}/scene.tif"))

    decoded = open_raster(written).gs.mask_and_scale()

    assert decoded.red.dtype == np.float32


def test_crs_name_stays_short_however_the_crs_was_built() -> None:
    coded = raster({"red": (("y", "x"), np.zeros((8, 8), "uint16"))}, geobox())
    named = raster(
        {
            "red": (
                ("y", "x"),
                np.zeros((8, 8), "uint16"),
            )
        },
        geobox(crs="+proj=laea +lat_0=52 +lon_0=10"),
    )

    assert coded.gs.crs_name == "EPSG:32633"
    assert coded.red.gs.crs_name == "EPSG:32633"
    # A CRS read back off spatial_ref prints its whole WKT, which this refuses to.
    assert len(str(coded.gs.crs)) > 100
    assert named.gs.crs_name == "unknown"
