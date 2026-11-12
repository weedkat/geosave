"""Several saved rasters read as one."""

from __future__ import annotations

from tests.conftest import write_cogs

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import xarray as xr

from geosave.geodata import open_vector
from geosave.geodata.io.formats import geotiff
from geosave.geodata.io.openers import open_raster
from tests.geodata.conftest import build_raster

from shapely import from_wkt
from odc.geo.geom import Geometry
from geosave.geodata import io

import zarr
from geosave.geodata.io.formats import netcdf
from geosave.geodata.io.formats import zarr as zarr_io
from geosave.geodata import attrs


def _flat():
    return build_raster(times=1).squeeze("time", drop=False)


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".parquet"])
def test_a_plain_vector_reads_back_with_its_own_columns(tmp_path, suffix):
    import geopandas as gpd
    from shapely.geometry import Point

    from geosave.geodata import open_vector

    plots = gpd.GeoDataFrame(
        {"class": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
    )
    path = tmp_path / f"plots{suffix}"
    writer = {
        ".geojson": plots.gs.to_geojson,
        ".gpkg": plots.gs.to_geopackage,
        ".parquet": plots.gs.to_geoparquet,
    }[suffix]

    read = open_vector(writer(path))

    assert sorted(read.columns) == ["class", "geometry"]


def test_a_signed_url_opens_by_the_suffix_of_its_path(monkeypatch) -> None:
    from geosave.geodata.io import openers

    opened = []
    monkeypatch.setattr(
        openers.rasterio, "open_raster", lambda source, **options: opened.append(source)
    )
    url = "https://host/scene/B02.tif?st=2026-10-11&sig=CXPh%3D"

    open_raster(url)

    assert opened == [url]


def test_a_signed_vector_url_preserves_its_query(tmp_path, monkeypatch) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    from geosave.geodata.io import openers

    url = "https://host/catalog.parquet?sig=abc%3D"
    path = tmp_path / "catalog.parquet"
    gpd.GeoDataFrame(
        {"class": [2]}, geometry=[Point(1, 3)], crs="EPSG:4326"
    ).to_parquet(path)
    original = openers.geoparquet.storage.location

    # Replace remote access while keeping dispatch and GeoParquet decoding real.
    def local(source, storage_options=None):
        assert source == url
        return original(path, storage_options)

    monkeypatch.setattr(openers.geoparquet.storage, "location", local)
    restored = open_vector(url)

    assert restored["class"].tolist() == [2]
    assert restored.geometry.iloc[0] == Point(1, 3)


def test_a_read_geotiff_names_the_file_it_came_from(tmp_path: Path) -> None:
    (path,) = write_cogs(build_raster(), tmp_path, format_paths="scene.tif")

    assert open_raster(path).encoding["source"] == str(path)


@pytest.mark.parametrize("georeferenced", [True, False])
def test_asset_reference_is_not_automatically_stac(tmp_path, georeferenced):
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {
            "id": ["tile"],
            "assets": [{"image": {"href": str(tmp_path / "image.nc")}}],
            "row_off": [0],
            "col_off": [0],
            "height": [1],
            "width": [1],
        },
        geometry=[box(0, 0, 1, 1) if georeferenced else None],
        crs="EPSG:4326" if georeferenced else None,
    )
    path = tmp_path / "reference.parquet"
    frame.gs.to_geoparquet(path)
    assert b"stac-geoparquet" not in pq.read_schema(path).metadata
    # A plain table's columns are stored as given, hrefs included.
    stored = pq.read_table(path).column("assets")[0].as_py()
    assert stored["image"]["href"] == str(tmp_path / "image.nc")
    restored = open_vector(path)
    assert restored.crs == frame.crs
    assert restored.iloc[0].assets["image"]["href"] == str(tmp_path / "image.nc")


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_unindexed_pixel_raster_needs_no_geographic_id(tmp_path, format):
    import xarray as xr

    from geosave.geodata import open_raster

    source = xr.Dataset({"image": (("y", "x"), np.arange(4).reshape(2, 2))})
    suffix = ".nc" if format == "netcdf" else ".zarr"
    saved = getattr(source.gs, f"to_{format}")(tmp_path / f"pixels{suffix}")
    with open_raster(saved) as restored:
        xr.testing.assert_equal(restored, source)


def test_open_vector_does_not_read_a_folder(rows, tmp_path) -> None:
    rows.gs.to_geoparquet(tmp_path / "catalog" / "items.parquet")

    with pytest.raises(ValueError, match="no supported vector format"):
        open_vector(tmp_path / "catalog")


@pytest.mark.parametrize("suffix", [".zarr", ".nc"])
@pytest.mark.parametrize("grouped", [False, True])
def test_stored_hierarchy_determines_type_without_computing_pixels(
    tmp_path, suffix, grouped
):
    import dask
    import dask.array as da

    source = build_raster(times=2)
    written = xr.DataTree.from_dict({"/scene": source}) if grouped else source
    path = tmp_path / f"scene{suffix}"
    if suffix == ".zarr":
        written.gs.to_zarr(path)
    else:
        written.gs.to_netcdf(path)

    def refuse(*args, **kwargs):
        raise AssertionError("Opening computed pixels")

    with dask.config.set(scheduler=refuse):
        restored = open_raster(path)
    with restored:
        assert isinstance(restored, xr.DataTree if grouped else xr.Dataset)
        dataset = restored["scene"].to_dataset() if grouped else restored
        assert isinstance(dataset.red.data, da.Array)
        xr.testing.assert_equal(dataset, source)


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".parquet"])
@pytest.mark.parametrize("columns", [[], ["class"]])
def test_vector_column_selection_retains_geometry_and_filters_bbox(
    tmp_path, suffix, columns
):
    import geopandas as gpd
    from shapely.geometry import Point

    features = gpd.GeoDataFrame(
        {"class": [1, 2], "other": [3, 4]},
        geometry=[Point(0, 0), Point(10, 10)],
        crs=4326,
    )
    writer = {
        ".geojson": features.gs.to_geojson,
        ".gpkg": features.gs.to_geopackage,
        ".parquet": features.gs.to_geoparquet,
    }[suffix]
    kwargs = {"write_covering_bbox": True} if suffix == ".parquet" else {}
    saved = writer(tmp_path / f"features{suffix}", **kwargs)
    result = open_vector(saved, columns=columns, bbox=(-1, -1, 1, 1))
    assert list(result.columns) == [*columns, "geometry"]
    assert result.geometry.tolist() == [Point(0, 0)]
    assert result.crs == features.crs


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".parquet", ".tif"])
def test_remote_filesystem_options_work_for_real_reads(tmp_path, bucket, suffix):
    import geopandas as gpd
    from shapely.geometry import Point
    from upath import UPath

    destination = UPath(f"{bucket}/remote{suffix}", skip_instance_cache=True)
    if suffix == ".tif":
        source = _flat()
        local = geotiff.write_cog(source, tmp_path / "scene.tif")
    else:
        source = gpd.GeoDataFrame({"class": [2]}, geometry=[Point(1, 3)], crs=4326)
        writer = {
            ".geojson": source.gs.to_geojson,
            ".gpkg": source.gs.to_geopackage,
            ".parquet": source.gs.to_geoparquet,
        }[suffix]
        local = writer(tmp_path / f"scene{suffix}")
    destination.write_bytes(local.read_bytes())
    if suffix == ".tif":
        with open_raster(destination) as restored:
            np.testing.assert_array_equal(restored.red.compute(), source.red)
    else:
        restored = open_vector(
            str(destination), storage_options={"skip_instance_cache": True}
        )
        assert restored.geometry.tolist() == source.geometry.tolist()
        assert restored["class"].tolist() == [2]


def test_hugging_face_repository_raster_uses_explicit_filesystem_options(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from geosave.geodata.io import rasterio as reader

    source = _flat()
    local = geotiff.write_cog(source, tmp_path / "scene.tif")

    @contextmanager
    def download(path):
        assert path.storage_options["token"] == "test-token"
        yield local

    monkeypatch.setattr(reader, "localize", download)
    with open_raster(
        "hf://datasets/owner/repo/scene.tif", storage_options={"token": "test-token"}
    ) as restored:
        np.testing.assert_array_equal(restored.red.compute(), source.red)


def test_remote_raster_download_lives_until_close_and_is_removed_on_failure(
    bucket, tmp_path, monkeypatch
):
    from tempfile import TemporaryDirectory
    from geosave.geodata.io import storage

    folders = []

    def temporary():
        folder = TemporaryDirectory()
        folders.append(Path(folder.name))
        return folder

    local = geotiff.write_cog(_flat(), tmp_path / "scene.tif")
    remote = storage.location(f"{bucket}/scene.tif")
    remote.write_bytes(local.read_bytes())
    monkeypatch.setattr(storage, "TemporaryDirectory", temporary)
    restored = open_raster(remote)
    assert folders[0].exists()
    np.testing.assert_array_equal(restored.red.compute(), _flat().red)
    restored.close()
    restored.close()
    assert not folders[0].exists()
    with pytest.raises(FileNotFoundError):
        open_raster(f"{bucket}/missing.tif")
    assert not folders[1].exists()


@pytest.mark.parametrize("suffix", ["nc", "zarr", "tif"])
def test_extent_round_trip(tmp_path, suffix):
    geometry = Geometry(
        from_wkt(
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0), "
            "(1 1, 1 2, 2 2, 2 1, 1 1)), ((5 0, 6 0, 5 1, 5 0)))"
        ),
        "EPSG:3857",
    )
    raster = build_raster().gs.write_extent(geometry)
    path = tmp_path / f"raster.{suffix}"
    if suffix == "nc":
        io.formats.netcdf.write_netcdf(raster, path)
    elif suffix == "zarr":
        io.formats.zarr.write_zarr(raster, path)
    else:
        io.formats.geotiff.write_cog(raster, path)
    with io.open_raster(path) as restored:
        assert restored.gs.region.extent == geometry
        assert restored.red.gs.region.extent == geometry
        np.testing.assert_array_equal(restored.red.values, raster.red.values)


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [
        ("tif", geotiff.write_cog),
        ("nc", netcdf.write_netcdf),
        ("zarr", zarr_io.write_zarr),
    ],
)
@pytest.mark.parametrize("decoded", [False, True])
def test_both_representations_round_trip_per_variable_packing(
    tmp_path, suffix, writer, decoded
):
    source = build_raster().gs.write_nodata(65535)
    source = source.gs.write_packing(
        attrs.CFPacking(fill_value=65535, scale_factor=0.0001, add_offset=-0.1),
        target=["red"],
    )
    source = source.gs.write_packing(
        attrs.CFPacking(fill_value=65535, scale_factor=0.0002, add_offset=-0.2),
        target=["nir"],
    )
    for name in source.data_vars:
        source[name].data = np.array([[0, 2000], [3000, 65535]], "uint16")
    expected = source.gs.mask_and_scale()
    written = expected if decoded else source
    path = tmp_path / f"scene.{suffix}"

    writer(written.chunk({"x": 1, "y": 1}), path)

    with io.open_raster(path, mask_and_scale=False) as restored:
        assert restored.gs.geobox == source.gs.geobox
        for name in source.data_vars:
            assert restored[name].dtype == source[name].dtype
            np.testing.assert_array_equal(restored[name], source[name])
            assert "scale_factor" not in restored[name].encoding
            assert (
                restored[name].attrs["scale_factor"]
                == source[name].attrs["scale_factor"]
            )
            assert restored[name].attrs["_FillValue"] == 65535

    with io.open_raster(path, mask_and_scale=True) as restored:
        for name in source.data_vars:
            np.testing.assert_allclose(
                restored[name], expected[name], atol=1e-7, equal_nan=True
            )
            assert "scale_factor" not in restored[name].attrs
            assert (
                restored[name].encoding["scale_factor"]
                == source[name].attrs["scale_factor"]
            )
            assert restored[name].encoding["_FillValue"] == 65535

    # Rewriting an externally decoded read must also encode exactly once.
    rewritten = tmp_path / f"rewritten.{suffix}"
    with io.open_raster(path, mask_and_scale=True) as loaded:
        writer(loaded, rewritten)
    with io.open_raster(rewritten, mask_and_scale=False) as restored:
        np.testing.assert_array_equal(restored.red, source.red)

    if suffix == "zarr":
        assert zarr.open_group(path)["red"].fill_value == 65535
