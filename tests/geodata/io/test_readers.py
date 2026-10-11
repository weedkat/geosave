"""Several saved rasters read as one."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
from upath import UPath

from geosave_engine.geodata import io
from geosave_engine.geodata.io.readers import open_raster, open_stack
from geosave_engine.geodata.warnings import DroppedAttrsWarning
from geosave_engine.geodata.io import geotiff, netcdf, zarr

from tests.geodata.conftest import build_raster


def write_scenes(folder: Path, cube: xr.Dataset, *, split: bool = False) -> list[Path]:
    """Write one COG per scene, or per scene and band, under arbitrary names."""
    paths = []
    for position in range(cube.sizes["time"]):
        scene = cube.isel(time=position)
        parts = [scene[[name]] for name in scene.data_vars] if split else [scene]
        for part in parts:
            name = f"file{len(paths)}.tif"
            paths.append(geotiff.write_cog(part, folder / name))
    return paths


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
        open_stack(tmp_path)


def test_two_entries_naming_one_layer_are_refused(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    netcdf.write(_flat()[["nir"]], folder / "label.nc")
    zarr.write(_flat()[["nir"]], folder / "label.zarr")

    with pytest.raises(ValueError, match="'label'"):
        open_stack(folder)


def test_a_directory_of_zarr_layers_opens_one_group_per_store(tmp_path: Path) -> None:
    folder = tmp_path / "s1"
    folder.mkdir()
    zarr.write(_flat()[["nir"]], folder / "label.zarr")
    zarr.write(build_raster(times=2)[["red"]], folder / "optical.zarr")

    assert open_stack(folder).gs.groups == ("label", "optical")


@pytest.mark.parametrize("remote", [True, False], ids=["memory", "file-url"])
def test_a_folder_url_of_zarr_layers_opens_one_group_per_store(
    bucket: str, tmp_path: Path, remote: bool
) -> None:
    folder = f"{bucket}/s1" if remote else (tmp_path / "s1").as_uri()
    zarr.write(_flat()[["nir"]], f"{folder}/label.zarr")
    zarr.write(build_raster(times=2)[["red"]], f"{folder}/optical.zarr")

    assert open_stack(folder).gs.groups == ("label", "optical")


def test_open_raster_reads_a_remote_upath(bucket: str) -> None:
    store = UPath(bucket) / "scene.zarr"
    zarr.write(build_raster()[["red"]], store)

    assert open_raster(store).gs.variables == ("red",)


@pytest.mark.parametrize("fail", [False, True])
def test_directory_reader_owns_opened_layers(tmp_path, monkeypatch, fail):
    from geosave_engine.geodata.io import readers

    for name in ("a.zarr", "b.zarr"):
        (tmp_path / name).touch()
    closed = []

    def read(path, **options):
        if fail and path.stem == "b":
            raise ValueError("unreadable layer")
        raster = _flat().copy()
        raster.set_close(lambda: closed.append(path.stem))
        return raster

    monkeypatch.setattr(readers, "open_raster", read)
    if fail:
        with pytest.raises(ValueError, match="unreadable"):
            open_stack(tmp_path)
        assert closed == ["a"]
    else:
        sample = open_stack(tmp_path)
        assert closed == []
        sample.close()
        assert sorted(closed) == ["a", "b"]


def test_named_sources_open_one_group_each(tmp_path: Path) -> None:
    label = geotiff.write_cog(_flat()[["nir"]], tmp_path / "a.tif")
    optical = zarr.write(build_raster(times=2)[["red"]], tmp_path / "b.zarr")

    sample = open_stack({"label": label, "optical": optical})

    assert sample.gs.groups == ("label", "optical")
    assert sample.gs.rasters["optical"].sizes["time"] == 2


@pytest.mark.parametrize("suffix", [".geojson", ".gpkg", ".parquet"])
def test_a_plain_vector_reads_back_with_its_own_columns(tmp_path, suffix):
    import geopandas as gpd
    from shapely.geometry import Point

    from geosave_engine.geodata import open_vector

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

    import geosave_engine as gs

    plots = gpd.GeoDataFrame({"name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    path = plots.gs.to_geoparquet(tmp_path / "plots.parquet")

    assert gs.open_vector(path)["name"].tolist() == ["a"]
    assert not hasattr(gs.geodata, "read_vector")


def test_open_raster_takes_one_path_not_a_list(tmp_path: Path) -> None:
    (file,) = build_raster().gs.to_cog(f"{tmp_path}/scene.tif")

    with pytest.raises(TypeError, match="CogFile"):
        open_raster([file.href, file.href])


def test_a_tag_the_records_agree_on_survives(tmp_path: Path) -> None:
    cube = build_raster(times=2).assign_attrs(TIFFTAG_ARTIST="geosave")
    files = cube.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")

    assert open_raster(files).attrs["TIFFTAG_ARTIST"] == "geosave"


def test_a_tag_the_records_state_differently_drops_and_warns(tmp_path: Path) -> None:
    cube = build_raster(times=2)
    files = [
        file
        for position in range(2)
        for file in cube.isel(time=[position])
        .assign_attrs(TIFFTAG_ARTIST=f"artist{position}")
        .gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif")
    ]

    with pytest.warns(DroppedAttrsWarning, match="TIFFTAG_ARTIST"):
        restored = open_raster(files)

    assert "TIFFTAG_ARTIST" not in restored.attrs
