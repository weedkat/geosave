import asyncio
from pathlib import Path

import fsspec
import numpy as np
import pytest
import xarray as xr
from upath import UPath

from geosave.geodata import io
from geosave.geodata.io.storage import (
    is_local,
    local_path,
    location,
    open_geotiff,
    read_coarsest,
    write,
)

from tests.geodata.conftest import build_raster


def test_a_location_carries_its_storage_options() -> None:
    url = location("memory://catalogs/catalog.parquet", {"skip_instance_cache": True})

    assert url.protocol == "memory"
    assert url.storage_options == {"skip_instance_cache": True}


def test_a_location_merges_options_into_a_upath_it_is_given() -> None:
    given = UPath("memory://catalogs/catalog.parquet", skip_instance_cache=True)

    url = location(given, {"auto_mkdir": False})

    assert url.storage_options == {"skip_instance_cache": True, "auto_mkdir": False}


@pytest.mark.parametrize(
    ("path", "local"),
    [
        ("scene.tif", True),
        ("/data/scene.tif", True),
        ("file:///data/scene.tif", True),
        (UPath("/data/scene.tif"), True),
        ("memory://bucket/scene.tif", False),
        ("s3://bucket/scene.tif", False),
    ],
)
def test_is_local_reads_the_protocol(path, local: bool) -> None:
    assert is_local(path) is local


def test_a_local_path_drops_the_file_scheme() -> None:
    assert local_path("file:///data/scene.tif") == Path("/data/scene.tif")


@pytest.mark.parametrize("path", ["scene.tif", "file:///data/scene.tif"])
def test_a_local_location_is_a_pathlib_path(path) -> None:
    assert isinstance(location(path), Path)


def test_a_upath_destination_writes_like_a_url(bucket: str) -> None:
    written = write(
        UPath(f"{bucket}/scene.tif"),
        lambda target: target.write_bytes(b"pixels"),
        suffixes=(".tif",),
    )

    assert written == UPath(f"{bucket}/scene.tif")
    assert written.read_bytes() == b"pixels"


@pytest.mark.parametrize(
    ("name", "write"),
    [
        (
            "forest.tif",
            lambda cube, url: io.geotiff.write_cog(cube.isel(time=0), url).path,
        ),
        ("forest.zarr", lambda cube, url: io.zarr.write(cube, url).path),
        ("forest.nc", lambda cube, url: io.netcdf.write(cube, url).path),
    ],
)
def test_a_file_url_writes_to_the_local_path_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name, write
) -> None:
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    written = write(build_raster(times=1), f"file://{tmp_path}/{name}")

    assert written == tmp_path / name
    assert isinstance(written, UPath)
    assert (tmp_path / name).exists()
    assert list(elsewhere.iterdir()) == []


def test_a_local_write_is_staged_then_moved_whole(tmp_path: Path) -> None:
    staged = []

    def save(target: Path) -> None:
        staged.append(target)
        target.write_bytes(b"pixels")

    written = write(tmp_path / "scene.tif", save, suffixes=(".tif",))

    assert written == tmp_path / "scene.tif"
    assert written.read_bytes() == b"pixels"
    assert staged[0] != written
    assert staged[0].suffix == ".tif"
    assert list(tmp_path.iterdir()) == [written]


def test_a_failed_local_write_leaves_the_destination_untouched(tmp_path: Path) -> None:
    destination = tmp_path / "scene.tif"
    destination.write_bytes(b"first")

    def save(target: Path) -> None:
        target.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write(destination, save, suffixes=(".tif",), overwrite=True)

    assert destination.read_bytes() == b"first"
    assert list(tmp_path.iterdir()) == [destination]


def test_a_local_write_creates_missing_folders(tmp_path: Path) -> None:
    written = write(
        tmp_path / "new" / "scene.tif",
        lambda target: target.write_bytes(b"pixels"),
        suffixes=(".tif",),
    )

    assert written.read_bytes() == b"pixels"


def test_an_existing_local_file_refuses_without_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "scene.tif"
    destination.write_bytes(b"first")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        write(
            destination, lambda target: pytest.fail("must not run"), suffixes=(".tif",)
        )


def test_a_wrong_suffix_refuses_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in one of"):
        write(
            tmp_path / "scene.png",
            lambda target: pytest.fail("must not run"),
            suffixes=(".tif", ".tiff"),
        )


def test_a_remote_write_uploads_what_was_written(bucket: str) -> None:
    written = write(
        f"{bucket}/scene.tif",
        lambda target: target.write_bytes(b"pixels"),
        suffixes=(".tif",),
    )

    assert written == UPath(f"{bucket}/scene.tif")
    assert fsspec.open(f"{bucket}/scene.tif").open().read() == b"pixels"


def test_a_failed_remote_write_uploads_nothing(bucket: str) -> None:
    def save(target: Path) -> None:
        target.write_bytes(b"partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        write(f"{bucket}/scene.tif", save, suffixes=(".tif",))

    assert not fsspec.filesystem("memory").exists(f"{bucket}/scene.tif")


def test_an_existing_remote_object_refuses_without_overwrite(bucket: str) -> None:
    url = f"{bucket}/scene.tif"
    write(url, lambda target: target.write_bytes(b"first"), suffixes=(".tif",))

    with pytest.raises(FileExistsError):
        write(url, lambda target: pytest.fail("must not run"), suffixes=(".tif",))
    write(
        url,
        lambda target: target.write_bytes(b"second"),
        suffixes=(".tif",),
        overwrite=True,
    )

    assert fsspec.open(url).open().read() == b"second"


def test_a_remote_write_returns_a_upath_carrying_its_options(bucket: str) -> None:
    written = write(
        f"{bucket}/scene.tif",
        lambda target: target.write_bytes(b"pixels"),
        suffixes=(".tif",),
        storage_options={"skip_instance_cache": True},
    )

    assert written.storage_options == {"skip_instance_cache": True}


def test_storage_options_cannot_choose_the_filesystem() -> None:
    with pytest.raises(ValueError, match="protocol"):
        location("memory://catalogs/catalog.parquet", {"protocol": "s3"})  # type: ignore[typeddict-unknown-key]



def write_cog(path: Path) -> Path:
    band = xr.DataArray(
        np.arange(256 * 256, dtype="uint16").reshape(256, 256),
        dims=("y", "x"),
        coords={
            "y": 9_000_000 - 10 * np.arange(256),
            "x": 500_000 + 10 * np.arange(256),
        },
    ).rio.write_crs("EPSG:32749")
    band.rio.to_raster(path, driver="COG")
    return path


def test_open_geotiff_reads_a_local_cog(tmp_path) -> None:
    geotiff = open_geotiff(write_cog(tmp_path / "band.tif"))

    assert (geotiff.height, geotiff.width) == (256, 256)
    assert geotiff.crs.to_epsg() == 32749


def test_open_geotiff_opens_inside_a_running_event_loop(tmp_path) -> None:
    path = write_cog(tmp_path / "band.tif")

    async def open_in_loop():
        return open_geotiff(path)

    assert asyncio.run(open_in_loop()).width == 256


def test_read_coarsest_reads_the_smallest_overview(tmp_path) -> None:
    values = read_coarsest(open_geotiff(write_cog(tmp_path / "band.tif")))

    assert values.ndim == 3 and values.shape[0] == 1
    assert values.shape[1] <= 256


def test_a_web_url_reads_over_http_and_a_bucket_unsigned_on_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from obstore.store import HTTPStore, S3Store

    from geosave.geodata.io.storage import create_store

    web = create_store("https://sentinel-cogs.s3.us-west-2.amazonaws.com/a/b")
    assert isinstance(web, HTTPStore)
    monkeypatch.setenv("AWS_NO_SIGN_REQUEST", "YES")
    bucket = create_store("s3://sentinel-cogs/a", {"region": "us-west-2"})
    assert isinstance(bucket, S3Store)
    assert str(bucket.config.get("skip_signature")).lower() == "true"
