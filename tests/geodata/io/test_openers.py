"""Several saved rasters read as one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from upath import UPath

from geosave.geodata import io, open_vector
from geosave.geodata.io import geotiff, netcdf, zarr
from geosave.geodata.io.openers import open_raster, open_tree
from geosave.geodata.warnings import DroppedAttrsWarning
from tests.geodata.conftest import build_raster
from tests.geodata.conftest import flat


def test_a_tiff_on_an_hf_bucket_opens_through_the_s3_gateway(monkeypatch) -> None:
    opened: list[str] = []

    def open_(path, *args, **kwargs):
        opened.append(str(path))
        raise OSError("no network in this test")

    monkeypatch.setattr(io.gdal.rasterio, "open", open_)

    with pytest.raises(OSError, match="no network"):
        open_raster("hf://buckets/me/samples/forest/a.tif")
    assert opened == ["s3://me/samples/forest/a.tif"]


def _flat():
    return build_raster(times=1).squeeze("time", drop=False)


def test_a_directory_holding_no_raster_is_refused(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("nothing here")

    with pytest.raises(ValueError, match="holds no layer"):
        open_tree(tmp_path)


def test_two_entries_naming_one_layer_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    netcdf.write(_flat()[["nir"]], folder / "label.nc")
    zarr.write(_flat()[["nir"]], folder / "label.zarr")

    with pytest.raises(ValueError, match="'label'"):
        open_tree(folder)


def test_a_directory_of_zarr_layers_opens_one_group_per_store(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    zarr.write(_flat()[["nir"]], folder / "label.zarr")
    zarr.write(build_raster(times=2)[["red"]], folder / "optical.zarr")

    assert open_tree(folder).gs.groups == ("label", "optical")


def test_storage_options_reach_a_folder_of_cog_and_zarr_layers(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    geotiff.write_cog(_flat()[["red"]], folder / "optical.tif")
    zarr.write(_flat()[["nir"]], folder / "label.zarr")

    tree = open_tree(folder, storage_options={})

    assert tree.gs.groups == ("label", "optical")


@pytest.mark.parametrize("remote", [True, False], ids=["memory", "file-url"])
def test_a_folder_url_of_zarr_layers_opens_one_group_per_store(
    bucket: str, tmp_path: Path, remote: bool
) -> None:
    folder = f"{bucket}/s1" if remote else (tmp_path / "s1").as_uri()
    zarr.write(_flat()[["nir"]], f"{folder}/label.zarr")
    zarr.write(build_raster(times=2)[["red"]], f"{folder}/optical.zarr")

    assert open_tree(folder).gs.groups == ("label", "optical")


def test_open_raster_reads_a_remote_upath(bucket: str) -> None:
    store = UPath(bucket) / "scene.zarr"
    zarr.write(build_raster()[["red"]], store)

    assert open_raster(store).gs.variables == ("red",)


@pytest.mark.parametrize("fail", [False, True])
def test_directory_reader_owns_opened_layers(tmp_path, monkeypatch, fail):
    from geosave.geodata.io import openers

    for name in ("a.zarr", "b.zarr"):
        (tmp_path / name).touch()
    closed = []

    def read(path, **options):
        if fail and path.stem == "b":
            raise ValueError("unreadable layer")
        raster = _flat().copy()
        raster.set_close(lambda: closed.append(path.stem))
        return raster

    monkeypatch.setattr(openers, "open_raster", read)
    if fail:
        with pytest.raises(ValueError, match="unreadable"):
            open_tree(tmp_path)
        assert closed == ["a"]
    else:
        sample = open_tree(tmp_path)
        assert closed == []
        sample.close()
        assert sorted(closed) == ["a", "b"]


def test_named_sources_open_one_group_each(tmp_path: Path) -> None:
    label = geotiff.write_cog(_flat()[["nir"]], tmp_path / "a.tif")
    optical = zarr.write(build_raster(times=2)[["red"]], tmp_path / "b.zarr")

    sample = open_tree({"label": label, "optical": optical})

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].sizes["time"] == 2


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


def test_vectors_open_through_open_vector_and_read_vector_is_gone(tmp_path) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    import geosave as gs

    plots = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    path = plots.gs.to_geoparquet(tmp_path / "plots.parquet")

    assert gs.open_vector(path)["name"].tolist() == ["a"]
    assert not hasattr(gs.geodata, "read_vector")


def test_open_raster_reads_back_what_to_cog_wrote(tmp_path: Path) -> None:
    import numpy as np

    scene = build_raster(times=2)
    paths = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif"))

    restored = open_raster(paths)

    assert restored.gs.variables == ("red", "nir")
    np.testing.assert_array_equal(restored.red.values, scene.red.values)


def test_open_tree_groups_scenes_by_their_collection(tmp_path: Path) -> None:
    from geosave.geodata import tree

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    scenes = tree({"optical": image, "label": label}).gs.to_cog(
        {
            "optical": f"{tmp_path}/optical_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/label.tif",
        }
    )

    assert open_tree(scenes).gs.groups == ("optical", "label")


def test_open_tree_refuses_scenes_naming_no_collection(tmp_path: Path) -> None:
    scenes = build_raster(times=2).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    with pytest.raises(ValueError, match="names no collection"):
        open_tree(scenes)


def test_a_tag_the_records_agree_on_survives(tmp_path: Path) -> None:
    cube = build_raster(times=2).assign_attrs(TIFFTAG_ARTIST="geosave")
    files = flat(cube.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif"))

    assert open_raster(files).attrs["TIFFTAG_ARTIST"] == "geosave"


def test_a_tag_the_records_state_differently_drops_and_warns(tmp_path: Path) -> None:
    cube = build_raster(times=2)
    files = [
        file
        for position in range(2)
        for file in flat(
            cube.isel(time=[position])
            .assign_attrs(TIFFTAG_ARTIST=f"artist{position}")
            .gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
        )
    ]

    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_ARTIST"):
        restored = open_raster(files)

    assert "TIFFTAG_ARTIST" not in restored.attrs


def test_a_signed_url_opens_by_the_suffix_of_its_path(monkeypatch) -> None:
    from geosave.geodata.io import openers

    opened = []
    monkeypatch.setattr(
        openers.gdal, "open_raster", lambda source, **options: opened.append(source)
    )
    url = "https://host/scene/B02.tif?st=2026-10-11&sig=CXPh%3D"

    open_raster(url)

    assert opened == [url]


def test_a_read_geotiff_names_the_file_it_came_from(tmp_path: Path) -> None:
    (path,) = flat(build_raster().gs.to_cog(tmp_path / "scene.tif"))

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
