import os
from pathlib import Path

import numpy as np
import pytest

from geosave.geodata.io.rasterio import configure_gdal, hf_to_s3


def test_gdal_configuration_sets_supplied_values(monkeypatch) -> None:
    monkeypatch.delenv("GDAL_HTTP_MAX_RETRY", raising=False)
    monkeypatch.delenv("AWS_NO_SIGN_REQUEST", raising=False)

    configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)

    assert os.environ["AWS_NO_SIGN_REQUEST"] == "TRUE"
    assert os.environ["GDAL_HTTP_MAX_RETRY"] == "3"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("hf://buckets/me/samples/forest/a.tif", "s3://me/samples/forest/a.tif"),
        ("s3://bucket/forest/a.tif", "s3://bucket/forest/a.tif"),
        ("https://example.com/a.tif", "https://example.com/a.tif"),
        ("/data/a.tif", "/data/a.tif"),
    ],
)
def test_gdal_reads_an_hf_bucket_through_its_s3_gateway(location, expected) -> None:
    assert hf_to_s3(location) == expected


def test_only_hf_buckets_have_an_s3_form() -> None:
    with pytest.raises(ValueError, match="buckets"):
        hf_to_s3("hf://datasets/me/repo/a.tif")


def _tiff(path, *, tiled: bool) -> str:
    import rasterio
    from rasterio.transform import from_origin

    options = {"tiled": True, "blockxsize": 256, "blockysize": 256} if tiled else {}
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=1,
        dtype="uint16",
        crs="EPSG:32749",
        transform=from_origin(500_000.0, 9_010_240.0, 10, 10),
        **options,
    ) as tiff:
        tiff.write(np.zeros((1, 1024, 1024), "uint16"))
    return str(path)


def test_lazy_chunks_follow_the_files_own_tiles(tmp_path) -> None:
    # A window then reads only the tiles it overlaps, not the whole band.
    from geosave.geodata.io.rasterio import open_raster

    band = open_raster(_tiff(tmp_path / "tiled.tif", tiled=True), chunks={}).band_1

    assert band.data.chunksize == (256, 256)


def test_a_striped_file_stays_one_chunk(tmp_path) -> None:
    from geosave.geodata.io.rasterio import open_raster

    band = open_raster(_tiff(tmp_path / "striped.tif", tiled=False), chunks={}).band_1

    assert band.data.chunksize == (1024, 1024)


def _refuse_to_open(monkeypatch) -> list:
    """Record what rasterio is asked to open, opening nothing."""
    import rasterio

    opened = []

    def spy(fp, *args, **kwargs):
        opened.append(str(fp))
        raise rasterio.errors.RasterioIOError("no network in tests")

    monkeypatch.setattr(rasterio, "open", spy)
    return opened


def test_an_hf_bucket_without_options_keeps_its_s3_gateway(monkeypatch) -> None:
    import rasterio

    from geosave.geodata.io.rasterio import open_raster

    opened = _refuse_to_open(monkeypatch)

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster("hf://buckets/me/samples/a.tif")

    assert opened == ["s3://me/samples/a.tif"]


def test_each_band_reads_its_own_pixels(tmp_path: Path) -> None:
    import xarray as xr

    from geosave.geodata.io.rasterio import open_raster

    values = np.stack([np.full((4, 4), k, "uint16") for k in (10, 20, 30)])
    cube = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={
            "band": [1, 2, 3],
            "y": 9_000_000 - 10 * np.arange(4),
            "x": 500_000 + 10 * np.arange(4),
        },
    ).rio.write_crs("EPSG:32749")
    cube.rio.to_raster(tmp_path / "three.tif")

    opened = open_raster(tmp_path / "three.tif")

    for index, variable in enumerate(opened.data_vars.values()):
        np.testing.assert_array_equal(variable.values, values[index])
