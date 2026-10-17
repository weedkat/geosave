"""Files and records read back as one raster."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from upath import UPath

from geosave.geodata.io import assemble
from geosave.geodata.io.raster.file import RasterFile
from geosave.geodata.warnings import DroppedAttrsWarning
from tests.geodata.conftest import build_raster, flat


def test_records_read_back_as_the_raster_written(tmp_path: Path) -> None:
    scene = build_raster(times=2)
    files = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}_{{band}}.tif"))

    restored = assemble.open_raster(files)

    assert list(restored.data_vars) == ["red", "nir"]
    assert restored.sizes["time"] == 2
    np.testing.assert_array_equal(restored.red.values, scene.red.values)


def test_dated_files_join_without_dropping_attrs(tmp_path: Path) -> None:
    files = flat(build_raster(times=2).gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DroppedAttrsWarning)
        restored = assemble.open_raster(files)

    assert restored.sizes["time"] == 2


def test_a_tree_reads_back_from_its_records(tmp_path: Path) -> None:
    from geosave.geodata import open_tree, tree

    image = build_raster(times=1)
    label = image.isel(time=0, drop=True)[["red"]].rename(red="label")
    scenes = tree({"optical": image, "label": label}).gs.to_cog(
        {
            "optical": f"{tmp_path}/{{collection}}_{{time:%Y%m%d}}.tif",
            "label": f"{tmp_path}/{{collection}}.tif",
        }
    )

    restored = open_tree(scenes)

    assert restored.gs.groups == ("optical", "label")
    assert "time" not in restored["label"].dims


def test_records_read_onto_a_grid_stay_lazy(tmp_path: Path) -> None:
    import dask

    scene = build_raster(times=2)
    files = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif"))
    grid = scene.gs.geobox

    read = assemble.open_raster(files, geobox=grid)

    assert dask.is_dask_collection(read.red)
    assert read.gs.geobox == grid


def test_a_grid_inside_the_files_own_is_sliced_not_warped(tmp_path: Path) -> None:
    import warnings

    from geosave.geodata.warnings import AssumedFillWarning

    scene = build_raster(times=2)
    files = flat(scene.gs.to_cog(f"{tmp_path}/s_{{time:%Y%m%d}}.tif"))
    window = scene.gs.geobox[1:2, 0:2]

    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumedFillWarning)
        read = assemble.open_raster(files, geobox=window)

    assert read.gs.geobox == window
    np.testing.assert_array_equal(read.red.values, scene.red.values[:, 1:2, 0:2])


def test_a_file_placed_by_control_points_is_warped_not_sliced(tmp_path: Path) -> None:
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS

    from geosave.geodata.io import gdal

    source = tmp_path / "gcp.tif"
    corners = [
        GroundControlPoint(row=0, col=0, x=3e5, y=5e6),
        GroundControlPoint(row=8, col=8, x=3e5 + 80, y=5e6 - 80),
        GroundControlPoint(row=0, col=8, x=3e5 + 80, y=5e6),
    ]
    with rasterio.open(
        source, "w", driver="GTiff", width=8, height=8, count=1, dtype="uint8"
    ) as dst:
        dst.write(np.ones((1, 8, 8), "uint8"))
        dst.gcps = (corners, CRS.from_epsg(32633))
    placed = gdal.open_raster(source).gs.geobox

    # A control-point grid has no affine transform to slice by.
    assert assemble._window(placed, build_raster().gs.geobox) is None


def _unnamed_tiff(
    path: Path, resolution: float, value: int, nodata: int | None = 0
) -> str:
    """Write a one-band GeoTIFF with no band description, as STAC hosts serve them."""
    from rasterio.transform import from_origin

    size = round(40 / resolution)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="uint16",
        crs="EPSG:32749",
        transform=from_origin(500_000.0, 9_000_040.0, resolution, resolution),
        nodata=nodata,
    ) as tiff:
        tiff.write(np.full((1, size, size), value, dtype="uint16"))
    return str(path)


def _named(path: str, *names: str) -> RasterFile:
    """Describe a file by a header naming its bands, as a catalog's Item does."""
    header = xr.Dataset({name: xr.DataArray(0) for name in names})
    return RasterFile(UPath(path), header)


def test_records_name_the_bands_a_file_leaves_unnamed(tmp_path: Path) -> None:
    files = [
        _named(_unnamed_tiff(tmp_path / f"{band}.tif", 10, value), band)
        for band, value in (("B02", 2), ("B03", 3))
    ]

    read = assemble.open_raster(files)

    assert list(read.data_vars) == ["B02", "B03"]
    assert int(read.B03.max()) == 3


def test_bands_of_mixed_resolution_read_onto_one_grid(tmp_path: Path) -> None:
    from odc.geo.geobox import GeoBox

    files = [
        _named(_unnamed_tiff(tmp_path / "B02.tif", 10, 2), "B02"),
        _named(_unnamed_tiff(tmp_path / "SCL.tif", 20, 4), "SCL"),
    ]
    window = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_040.0, 9_000_040.0),
        crs="EPSG:32749",
        resolution=10,
    )

    read = assemble.open_raster(files, geobox=window)

    assert list(read.data_vars) == ["B02", "SCL"]
    assert read.gs.geobox == window
    assert int(read.SCL.max()) == 4


def test_bands_without_fill_warn_once_per_read(tmp_path: Path) -> None:
    import warnings

    from affine import Affine
    from odc.geo.geobox import GeoBox

    from geosave.geodata.warnings import AssumedFillWarning

    files = [
        _named(_unnamed_tiff(tmp_path / f"{band}.tif", 10, 2, nodata=None), band)
        for band in ("B02", "B03")
    ]
    # Half a pixel off the files' own grid, so every band is warped.
    shifted = GeoBox(
        (3, 3), Affine(10, 0, 500_005.0, 0, -10, 9_000_035.0), "EPSG:32749"
    )

    messages = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assemble.open_raster(files, geobox=shifted)
        messages.append(
            [
                str(w.message)
                for w in caught
                if issubclass(w.category, AssumedFillWarning)
            ]
        )

    first, second = messages
    assert len(first) == 1
    assert "B02" in first[0] and "B03" in first[0]
    assert first == second


def test_one_name_over_several_bands_numbers_them(tmp_path: Path) -> None:
    from rasterio.transform import from_origin

    path = tmp_path / "visual.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32749",
        transform=from_origin(500_000.0, 9_000_020.0, 10, 10),
    ) as tiff:
        tiff.write(np.stack([np.full((2, 2), value, "uint8") for value in (1, 2, 3)]))

    read = assemble.open_raster([_named(str(path), "visual")])

    assert list(read.data_vars) == ["visual.1", "visual.2", "visual.3"]
    assert int(read["visual.3"].max()) == 3
