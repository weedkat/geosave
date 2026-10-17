import os
from pathlib import Path

import numpy as np
import pytest

from geosave.geodata.io.raster.gdal import configure_gdal, hf_to_s3
from tests.geodata.conftest import flat


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
        with memory.open(
            driver="GTiff", width=1, height=1, count=1, dtype="uint8"
        ) as dst:
            dst.write(np.array([[[1]]], dtype="uint8"))
            dst.set_band_description(1, "Red reflectance")
            dst.update_tags(1, variable_name="red", units="1")

        with memory.open() as src:
            _, bands = _attrs(src)

    assert list(bands) == ["red"]
    assert bands["red"]["long_name"] == "Red reflectance"
    assert bands["red"]["units"] == "1"
    assert bands["red"]["variable_name"] == "red"


@pytest.mark.parametrize(
    ("scale", "offset", "nodata", "expected"),
    [
        (1e-4, -0.1, 0, {"scale_factor": 1e-4, "add_offset": -0.1, "_FillValue": 0}),
        (1e-4, 0.0, None, {"scale_factor": 1e-4}),
        (1.0, 0.0, None, {}),
    ],
)
def test_band_packing_and_fill_read_as_their_cf_attrs(
    scale, offset, nodata, expected
) -> None:
    from rasterio.io import MemoryFile

    from geosave.geodata.io.raster.gdal import _attrs

    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", width=1, height=1, count=1, dtype="uint16", nodata=nodata
        ) as dst:
            dst.write(np.array([[[1]]], dtype="uint16"))
            dst.scales, dst.offsets = (scale,), (offset,)

        with memory.open() as src:
            _, bands = _attrs(src)

    keys = ("scale_factor", "add_offset", "_FillValue")
    assert {
        key: bands["band_1"][key] for key in keys if key in bands["band_1"]
    } == expected


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
    from geosave.geodata.io.raster.gdal import open_raster

    band = open_raster(_tiff(tmp_path / "tiled.tif", tiled=True), chunks={}).band_1

    assert band.data.chunksize == (256, 256)


def test_a_striped_file_stays_one_chunk(tmp_path) -> None:
    from geosave.geodata.io.raster.gdal import open_raster

    band = open_raster(_tiff(tmp_path / "striped.tif", tiled=False), chunks={}).band_1

    assert band.data.chunksize == (1024, 1024)


def _tiff_bytes(tmp_path) -> bytes:
    import geosave as gs
    from geosave.geodata import GeoAnchor

    anchor = GeoAnchor.from_coordinates(52, 13, 4, 10)
    band = (("y", "x"), np.full((4, 4), 7, "uint16"))
    (file,) = flat(
        gs.raster({"red": band}, anchor.geobox).gs.to_cog(tmp_path / "a.tif")
    )
    return Path(file).read_bytes()


def _refuse_to_open(monkeypatch) -> list:
    """Record what rasterio is asked to open, opening nothing."""
    import rasterio

    opened = []

    def spy(fp, *args, **kwargs):
        opened.append(str(fp))
        raise rasterio.errors.RasterioIOError("no network in tests")

    monkeypatch.setattr(rasterio, "open", spy)
    return opened


AWS_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_NO_SIGN_REQUEST",
    "AWS_S3_ENDPOINT",
    "AWS_HTTPS",
    "AWS_VIRTUAL_HOSTING",
)


@pytest.fixture
def clean_aws(monkeypatch):
    # Setting first registers each key with monkeypatch, so what a read sets is undone.
    for name in AWS_KEYS:
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)


def test_s3_storage_options_reach_gdal_as_its_config(monkeypatch, clean_aws) -> None:
    import rasterio

    from geosave.geodata.io.raster.gdal import open_raster

    opened = _refuse_to_open(monkeypatch)
    options = {
        "key": "minio",
        "secret": "secret",
        "client_kwargs": {"endpoint_url": "http://localhost:9000"},
    }

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster("s3://bucket/a.tif", storage_options=options)

    assert opened == ["s3://bucket/a.tif"]
    assert os.environ["AWS_ACCESS_KEY_ID"] == "minio"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert os.environ["AWS_S3_ENDPOINT"] == "localhost:9000"
    assert os.environ["AWS_HTTPS"] == "NO"
    assert os.environ["AWS_VIRTUAL_HOSTING"] == "FALSE"


def test_an_anonymous_s3_read_skips_signing(monkeypatch, clean_aws) -> None:
    import rasterio

    from geosave.geodata.io.raster.gdal import open_raster

    _refuse_to_open(monkeypatch)

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster("s3://bucket/a.tif", storage_options={"anon": True})

    assert os.environ["AWS_NO_SIGN_REQUEST"] == "TRUE"


def test_a_second_s3_read_replaces_what_the_first_set(monkeypatch, clean_aws) -> None:
    import rasterio

    from geosave.geodata.io.raster.gdal import open_raster

    _refuse_to_open(monkeypatch)
    minio = {"anon": True, "client_kwargs": {"endpoint_url": "http://localhost:9000"}}
    aws = {"key": "id", "secret": "secret"}

    for options in (minio, aws):
        with pytest.raises(rasterio.errors.RasterioIOError):
            open_raster("s3://bucket/a.tif", storage_options=options)

    assert os.environ["AWS_ACCESS_KEY_ID"] == "id"
    for stale in ("AWS_NO_SIGN_REQUEST", "AWS_S3_ENDPOINT", "AWS_HTTPS"):
        assert stale not in os.environ


def test_storage_options_for_another_service_are_left_to_its_reader(
    monkeypatch, clean_aws
) -> None:
    import rasterio

    from geosave.geodata.io.raster.gdal import open_raster

    opened = _refuse_to_open(monkeypatch)

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster("gs://bucket/a.tif", storage_options={"token": "anon"})

    assert opened == ["gs://bucket/a.tif"]
    assert not any(name in os.environ for name in AWS_KEYS)


def test_a_upath_carries_its_options_to_the_read(monkeypatch, clean_aws) -> None:
    import rasterio
    from upath import UPath

    from geosave.geodata.io.raster.gdal import open_raster

    _refuse_to_open(monkeypatch)

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster(UPath("s3://bucket/a.tif", key="id", secret="secret"))

    assert os.environ["AWS_ACCESS_KEY_ID"] == "id"


def test_a_local_file_ignores_no_options(tmp_path) -> None:
    from geosave.geodata.io.raster.gdal import open_raster

    path = tmp_path / "b.tif"
    path.write_bytes(_tiff_bytes(tmp_path))

    assert int(open_raster(path, storage_options={}).red.max()) == 7


def test_an_hf_bucket_without_options_keeps_its_s3_gateway(monkeypatch) -> None:
    import rasterio

    from geosave.geodata.io.raster.gdal import open_raster

    opened = _refuse_to_open(monkeypatch)

    with pytest.raises(rasterio.errors.RasterioIOError):
        open_raster("hf://buckets/me/samples/a.tif")

    assert opened == ["s3://me/samples/a.tif"]


def test_each_band_records_its_number_in_the_file(tmp_path: Path) -> None:
    import xarray as xr

    from geosave.geodata.io.raster.gdal import open_raster

    values = np.stack([np.full((4, 4), k, "uint16") for k in (10, 20, 30)])
    cube = xr.DataArray(
        values,
        dims=("band", "y", "x"),
        coords={"band": [1, 2, 3], "y": 9_000_000 - 10 * np.arange(4), "x": 500_000 + 10 * np.arange(4)},
    ).rio.write_crs("EPSG:32749")
    cube.rio.to_raster(tmp_path / "three.tif")

    opened = open_raster(tmp_path / "three.tif")

    assert [variable.encoding["band"] for variable in opened.data_vars.values()] == [1, 2, 3]
