"""Expand path formats into destinations without reading or writing pixels."""

import dask
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from geosave.geodata.io.paths import default_path_format, expand_paths
from tests.conftest import build_tree
from tests.geodata.conftest import build_raster

ROOT = UPath("memory://bucket/out")


def names(files: dict) -> list[str]:
    return [path.relative_to(ROOT).as_posix() for path in files]


def test_fields_select_dates_and_bands_without_computing():
    source = build_raster(times=2).chunk()

    def refuse(*args, **kwargs):
        raise AssertionError("path expansion computed pixels")

    with dask.config.set(scheduler=refuse):
        files = expand_paths(
            source, ROOT, "{collection}/{time:%Y%m%d}/{band}.tif", collection="s2"
        )
    assert names(files) == [
        "s2/20250601/red.tif",
        "s2/20250601/nir.tif",
        "s2/20250602/red.tif",
        "s2/20250602/nir.tif",
    ]
    selected = files[ROOT / "s2/20250602/red.tif"]
    assert list(selected.data_vars) == ["red"]
    assert selected.red.chunks is not None
    np.testing.assert_array_equal(selected.red.values, source.red.isel(time=1).values)
    assert pd.Timestamp(selected.time.values) == pd.Timestamp("2025-06-02")


def test_without_fields_the_dataset_stays_whole():
    source = build_raster(times=2)

    files = expand_paths(source, ROOT, "whole.any-format")

    assert files == {ROOT / "whole.any-format": source}


def test_band_splitting_preserves_observation_coordinates():
    source = build_raster(times=1).assign_coords(
        cloud_fraction=("time", [0.2]),
        time_bounds=(
            ("time", "edge"),
            np.array([["2025-06-01", "2025-06-30"]], dtype="datetime64[ns]"),
        ),
    )

    files = expand_paths(source, ROOT, "{time:%Y%m%d}/{band}.nc")

    for selected in files.values():
        assert selected.time_bounds.shape == (2,)
        assert selected.cloud_fraction.item() == 0.2


def test_colliding_dates_are_rejected_before_a_mapping_can_overwrite():
    with pytest.raises(ValueError, match="collid"):
        expand_paths(build_raster(times=2), ROOT, "{time:%Y}.tif")


@pytest.mark.parametrize(
    "path_format",
    [
        "/out/{band}.tif",
        "s3://bucket/{band}.tif",
        "../{band}.tif",
        "a/../../{band}.tif",
        "C:/out/{band}.tif",
        r"..\{band}.tif",
    ],
)
def test_a_path_format_cannot_replace_or_escape_the_folder(path_format):
    with pytest.raises(ValueError, match="relative|folder"):
        expand_paths(build_raster(), ROOT, path_format)


def test_a_band_name_cannot_escape_the_folder():
    source = build_raster().rename(red="../red")

    with pytest.raises(ValueError, match="relative|folder"):
        expand_paths(source, ROOT, "{band}.tif")


def test_the_folder_cannot_be_a_file():
    with pytest.raises(ValueError, match="path_format"):
        expand_paths(build_raster(), UPath("memory://bucket/scene.tif"))


@pytest.fixture
def mixed_tree():
    """s2 at 10 m and 20 m on two dates beside a timeless DEM on the 10 m grid."""
    from tests.geodata.core.raster.test_datatree import build_mixed_collection

    s2 = build_mixed_collection()
    dem = s2["s2/r10m"].to_dataset().isel(time=0, drop=True).rename(red="height")
    return build_tree({**s2.gs.rasters, "dem": dem})


def test_each_field_splits_its_axis(mixed_tree):
    files = expand_paths(
        mixed_tree,
        ROOT,
        {
            "s2": "{collection}/{res}/{time:%Y%m%d}/{band}.tif",
            "dem": "{collection}/{res}/{band}.tif",
        },
    )

    assert names(files) == [
        "s2/r10m/20240305/red.tif",
        "s2/r10m/20240405/red.tif",
        "s2/r20m/20240305/swir.tif",
        "s2/r20m/20240405/swir.tif",
        "dem/r10m/height.tif",
    ]
    assert list(files[ROOT / "s2/r20m/20240405/swir.tif"].data_vars) == ["swir"]


def test_a_collection_on_several_grids_needs_res_to_keep_them_apart(mixed_tree):
    with pytest.raises(ValueError, match=r"\{res\}"):
        expand_paths(mixed_tree, ROOT, {"s2": "s2/{time:%Y%m%d}.tif", "dem": "dem.tif"})


def test_several_collections_need_collection(mixed_tree):
    with pytest.raises(ValueError, match=r"\{collection\}"):
        expand_paths(mixed_tree, ROOT, "{res}.tif")


def test_a_path_format_per_collection(mixed_tree):
    files = expand_paths(
        mixed_tree, ROOT, {"s2": "s2/{res}_{time:%Y%m%d}.tif", "dem": "dem.tif"}
    )

    assert names(files) == [
        "s2/r10m_20240305.tif",
        "s2/r10m_20240405.tif",
        "s2/r20m_20240305.tif",
        "s2/r20m_20240405.tif",
        "dem.tif",
    ]


def test_a_mapping_names_every_collection(mixed_tree):
    with pytest.raises(ValueError, match="dem"):
        expand_paths(mixed_tree, ROOT, {"s2": "s2/{res}_{time:%Y%m%d}.tif"})


def test_the_default_lays_out_a_file_per_band_and_date(mixed_tree):
    files = expand_paths(mixed_tree, ROOT)

    assert names(files) == [
        "s2/20240305T000000/red.tif",
        "s2/20240405T000000/red.tif",
        "s2/20240305T000000/swir.tif",
        "s2/20240405T000000/swir.tif",
        "dem/height.tif",
    ]


def test_the_default_drops_the_levels_a_raster_lacks():
    dated = build_raster(times=1)
    timeless = build_raster()

    assert default_path_format(dated, None) == "{time:%Y%m%dT%H%M%S}/{band}.tif"
    assert default_path_format(timeless, None) == "{band}.tif"
    assert default_path_format(timeless, "dem") == "{collection}/{band}.tif"
    assert names(expand_paths(timeless, ROOT)) == ["red.tif", "nir.tif"]
