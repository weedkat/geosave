import os

import numpy as np
import pytest

from geosave.geodata.io.raster.gdal import configure_gdal, hf_to_s3


def test_gdal_configuration_sets_supplied_values(monkeypatch) -> None:
    monkeypatch.delenv("GDAL_HTTP_MAX_RETRY", raising=False)
    monkeypatch.delenv("AWS_NO_SIGN_REQUEST", raising=False)

    configure_gdal(aws_no_sign_request=True, gdal_http_max_retry=3)

    assert os.environ["AWS_NO_SIGN_REQUEST"] == "TRUE"
    assert os.environ["GDAL_HTTP_MAX_RETRY"] == "3"


def test_gdal_configuration_points_s3_at_another_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("AWS_VIRTUAL_HOSTING", raising=False)

    configure_gdal(aws_s3_endpoint="s3.hf.co", aws_virtual_hosting=False)

    assert os.environ["AWS_S3_ENDPOINT"] == "s3.hf.co"
    assert os.environ["AWS_VIRTUAL_HOSTING"] == "FALSE"


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


def test_gdal_configuration_reaches_a_plain_http_s3_service(monkeypatch) -> None:
    # Set first, so monkeypatch restores what configure_gdal writes.
    monkeypatch.setenv("AWS_HTTPS", "YES")
    monkeypatch.setenv("AWS_S3_ENDPOINT", "elsewhere")

    configure_gdal(aws_s3_endpoint="localhost:9000", aws_https=False)

    assert os.environ["AWS_HTTPS"] == "NO"
    assert os.environ["AWS_S3_ENDPOINT"] == "localhost:9000"


def test_gdal_configuration_can_skip_sidecar_probes(monkeypatch) -> None:
    # Set first, so monkeypatch restores what configure_gdal writes.
    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "FALSE")

    configure_gdal(gdal_disable_readdir_on_open="EMPTY_DIR")

    assert os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"



def test_band_slots_and_tags_read_as_one_variables_attrs() -> None:
    from rasterio.io import MemoryFile

    from geosave.geodata.io.raster.gdal import _attrs

    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=1, height=1, count=1, dtype="uint8") as dst:
            dst.write(np.array([[[1]]], dtype="uint8"))
            dst.set_band_description(1, "Red reflectance")
            dst.update_tags(1, variable_name="red", units="1")

        with memory.open() as src:
            _, bands = _attrs(src)

    assert list(bands) == ["red"]
    assert bands["red"]["long_name"] == "Red reflectance"
    assert bands["red"]["units"] == "1"
    assert bands["red"]["variable_name"] == "red"


def _tiff(path, *, tiled: bool) -> str:
    import rasterio
    from rasterio.transform import from_origin

    options = {"tiled": True, "blockxsize": 256, "blockysize": 256} if tiled else {}
    with rasterio.open(
        path, "w", driver="GTiff", width=1024, height=1024, count=1, dtype="uint16",
        crs="EPSG:32749", transform=from_origin(500_000.0, 9_010_240.0, 10, 10),
        **options,
    ) as tiff:
        tiff.write(np.zeros((1, 1024, 1024), "uint16"))
    return str(path)


def test_lazy_chunks_follow_the_files_own_tiles(tmp_path) -> None:
    # A window then reads only the tiles it overlaps, not the whole band.
    from geosave.geodata.io.raster.gdal import read

    band = read(_tiff(tmp_path / "tiled.tif", tiled=True), chunks={}).band_1

    assert band.data.chunksize == (256, 256)


def test_a_striped_file_stays_one_chunk(tmp_path) -> None:
    from geosave.geodata.io.raster.gdal import read

    band = read(_tiff(tmp_path / "striped.tif", tiled=False), chunks={}).band_1

    assert band.data.chunksize == (1024, 1024)
