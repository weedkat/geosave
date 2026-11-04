from datetime import datetime as dt
from typing import TYPE_CHECKING, assert_type

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geosave.geodata as gd
from geosave.geodata import GeoDataArray, GeoDataset, GeoDataTree
from tests.geodata.conftest import build_raster, flat
from tests.geodata.conftest import create_test_collection
from tests.conftest import build_tree


def test_to_catalog_writes_current_pixels_with_collection_time_and_band(tmp_path):
    import geopandas as gpd
    import geosave as gs

    original = build_raster(times=2)
    changed = original.copy()
    changed["red"] = changed.red + 1
    collection = create_test_collection("s2", description="Reflectance")
    rows = changed.gs.to_catalog(
        tmp_path,
        path_format="{collection}/{time:%Y%m%d}/{band}.tif",
        collection=collection,
    )
    assert isinstance(rows, gpd.GeoDataFrame)
    assert rows.collection.tolist() == ["s2", "s2"]
    assert rows.gs.collections["s2"].description == "Reflectance"
    assert (tmp_path / "s2/20250601/red.tif").is_file()
    assert (tmp_path / "s2/20250602/nir.tif").is_file()
    saved = rows.gs.to_geoparquet(tmp_path / "catalog.parquet")
    restored = gs.open_vector(saved).gs.to_datatree()
    raster = restored["s2"].gs.to_dataset()
    assert raster.red.chunks is not None
    np.testing.assert_array_equal(raster.red.values, changed.red.values)
    np.testing.assert_array_equal(raster.nir.values, changed.nir.values)
    assert raster.gs.geobox == changed.gs.geobox
    restored.close()


def test_to_catalog_preserves_tree_groups_on_their_native_grids(tmp_path):
    tree = build_tree(
        {"s2": build_raster(), "dem": build_raster(crs="EPSG:32633")}
    )
    rows = tree.gs.to_catalog(tmp_path, path_format="{collection}/{band}.tif")
    restored = rows.gs.to_datatree()
    assert restored.gs.groups == ("s2", "dem")
    assert restored["s2"].gs.to_dataset().gs.geobox == tree["s2"].gs.to_dataset().gs.geobox
    assert restored["dem"].gs.to_dataset().gs.geobox == tree["dem"].gs.to_dataset().gs.geobox
    restored.close()


def test_to_catalog_refuses_unsplit_dates_before_writing(tmp_path):
    with pytest.raises(ValueError, match="dates"):
        build_raster(times=2).gs.to_catalog(tmp_path, path_format="scene.tif", collection="s2")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_to_catalog_refuses_store_templates_before_writing(tmp_path, suffix):
    with pytest.raises(ValueError, match="to_zarr or to_netcdf"):
        build_raster(times=2).gs.to_catalog(
            tmp_path, path_format="{time:%Y%m%d}" + suffix, collection="s2"
        )
    assert not list(tmp_path.iterdir())


def test_to_catalog_dataarray_keeps_its_band_name_and_refuses_overwrite(tmp_path):
    source = build_raster(times=1).red
    rows = source.gs.to_catalog(tmp_path, path_format="{band}.tif", collection="s2")
    assert set(rows.iloc[0].assets) == {"red"}
    with pytest.raises(FileExistsError):
        source.gs.to_catalog(tmp_path, path_format="{band}.tif", collection="s2")


def test_to_catalog_refuses_unlocated_rasters_before_writing(tmp_path):
    source = xr.Dataset({"red": (("y", "x"), np.ones((2, 2), dtype="float32"))})
    with pytest.raises(ValueError, match="grid"):
        source.gs.to_catalog(tmp_path, path_format="scene.tif", collection="s2")
    assert not list(tmp_path.iterdir())


def test_to_catalog_refuses_a_store_beside_cogs_before_writing(tmp_path):
    tree = build_tree(
        {"s2": build_raster(), "dem": build_raster(crs="EPSG:32633")}
    )
    with pytest.raises(ValueError, match="to_zarr or to_netcdf"):
        tree.gs.to_catalog(tmp_path, path_format={"s2": "s2.tif", "dem": "dem.nc"})
    assert not list(tmp_path.iterdir())


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


def test_a_band_read_from_disk_decodes_to_float32(tmp_path) -> None:
    from geosave.geodata import open_raster

    (written,) = flat(build_raster(packed=True).gs.to_cog(tmp_path, path_format="scene.tif"))

    decoded = open_raster(written).gs.mask_and_scale()

    assert decoded.red.dtype == np.float32
