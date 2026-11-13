from pathlib import Path

import pytest
from upath import UPath
from upath.implementations.local import PosixUPath

from geosave.geodata.io.storage import check_destination, create_upath


def test_a_upath_merges_options_into_one_it_is_given() -> None:
    given = UPath("memory://catalogs/catalog.parquet", skip_instance_cache=True)

    created = create_upath(given, {"auto_mkdir": False})

    assert created.storage_options == {"skip_instance_cache": True, "auto_mkdir": False}


def test_a_file_url_is_a_local_path(tmp_path: Path) -> None:
    created = create_upath(f"file://{tmp_path}/scene.tif")

    assert isinstance(created, PosixUPath)
    assert created == tmp_path / "scene.tif"


def test_options_cannot_choose_another_filesystem() -> None:
    with pytest.raises(ValueError, match="protocol"):
        create_upath("s3://bucket/scene.tif", {"protocol": "memory"})  # type: ignore[typeddict-unknown-key]


@pytest.mark.parametrize("root", ["local", "memory"])
def test_a_destination_of_another_format_is_refused(root, tmp_path, bucket) -> None:
    folder = create_upath(tmp_path if root == "local" else bucket)

    with pytest.raises(ValueError, match=r"\.tif"):
        check_destination(folder / "scene.png", (".tif",), overwrite=False)


@pytest.mark.parametrize("root", ["local", "memory"])
def test_an_existing_destination_is_refused_unless_replaced(
    root, tmp_path, bucket
) -> None:
    folder = create_upath(tmp_path if root == "local" else bucket)
    existing = folder / "scene.tif"
    existing.parent.mkdir(parents=True, exist_ok=True)
    with existing.open("wb") as handle:
        handle.write(b"pixels")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        check_destination(existing, (".tif",), overwrite=False)
    check_destination(existing, (".tif",), overwrite=True)
