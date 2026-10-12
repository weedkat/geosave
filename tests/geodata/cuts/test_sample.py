"""A window row reads itself as an aligned lazy stack."""

from __future__ import annotations

import subprocess
import sys

import dask
import numpy as np
import pandas as pd
import pytest

from geosave.geodata import cuts


def test_a_window_reads_every_collection_on_its_grid(items, sample, grid) -> None:
    window = cuts.stacks(items, on="label").iloc[0]

    tree = window.gs.to_stack()

    assert sorted(tree.gs.groups) == ["dem", "label", "s1", "s2"]
    for raster in tree.gs.rasters.values():
        assert raster.gs.geobox == grid
    assert dask.is_dask_collection(tree["s2"].dataset.red)
    np.testing.assert_array_equal(
        tree["label"].dataset.label.values, sample["label"].dataset.label.values
    )


def test_a_row_states_the_grid_its_window_reads_onto(items, grid) -> None:
    windows = cuts.stacks(items, on="label")
    chip = cuts.chips(windows, (20, 30)).iloc[1]

    assert windows.iloc[0].gs.geobox == grid
    assert chip.gs.geobox == grid[0:20, 30:60]


def test_a_row_without_sources_has_no_sample_accessor() -> None:
    with pytest.raises(AttributeError, match="cuts table"):
        pd.Series({"a": 1}).gs


def test_a_reflect_halo_fills_only_past_the_parent(items, sample) -> None:
    chip = cuts.chips(cuts.stacks(items, on="label"), 32, overlap=8).iloc[0]

    label = chip.gs.to_stack()["label"].dataset.label.values

    (top, _), (left, _) = chip["pad"]
    source = sample["label"].dataset.label.values
    np.testing.assert_array_equal(label[top:, left:], source[: 32 - top, : 32 - left])
    np.testing.assert_array_equal(label[top - 1, left:], source[1, : 32 - left])


def test_a_saved_windows_table_reads_the_same_pixels(items, tmp_path) -> None:
    from geosave.geodata import open_vector

    cut = cuts.chips(cuts.stacks(items, on="label"), 32)
    path = cut.gs.to_geoparquet(tmp_path / "windows.parquet")

    back = open_vector(path)

    a = cut.iloc[3].gs.to_stack()["s2"].dataset.red.values
    b = back.iloc[3].gs.to_stack()["s2"].dataset.red.values
    np.testing.assert_array_equal(a, b)


def test_cuts_registers_the_row_accessor_on_import():
    code = (
        "import sys, pandas as pd, geosave; "
        "assert 'gs' not in dir(pd.Series); "
        "from geosave.geodata import cuts; "
        "assert 'gs' in dir(pd.Series); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_multiband_source_numbers_its_bands_by_asset(items, sample, tmp_path):
    import orjson
    import pystac

    from geosave.geodata import open_vector, stac

    import geosave.geodata.attrs as attrs

    optical = sample["s2"].dataset
    optical = optical.assign(
        red=attrs.stamp(optical.red, attrs.Spectral, attrs.Spectral(common_name="red")),
        nir=optical.red + 1,
    )
    owner = stac.create_collection("s2", description="Optical bands")
    files = optical.gs.to_cog(f"{tmp_path}/optical_{{time:%Y%m%d}}.tif")
    rows = stac.create_items(files, collection=owner)
    # A fresh optical collection with multiband assets, anchored by the label.
    combined = pd.concat(
        [items[items["collection"] == "label"], rows], ignore_index=True
    )
    path = combined.gs.to_geoparquet(tmp_path / "multiband.parquet")
    windows = cuts.stacks(open_vector(path), on="label")

    snapshots = orjson.loads(windows.iloc[0]["sources"])["s2"]
    records = stac.cog_files(pystac.Item.from_dict(snapshots[0]))
    assert records[0].bands == ("image.1", "image.2")

    recovered = windows.iloc[0].gs.to_stack()["s2"].dataset

    assert tuple(recovered.data_vars) == ("image.1", "image.2")
    assert dask.is_dask_collection(recovered["image.1"])
    assert recovered.gs.geobox == optical.gs.geobox
    np.testing.assert_array_equal(recovered.time.values, optical.time.values)
    np.testing.assert_array_equal(recovered["image.1"].values, optical.red.values)
    np.testing.assert_array_equal(recovered["image.2"].values, optical.nir.values)
    # The catalog's EO facts are metadata on the variable the asset key names.
    assert recovered["image.1"].gs.spectral.common_name == "red"


def test_catalog_eo_facts_reach_a_file_that_states_none(items, sample, tmp_path):
    from pystac.extensions.eo import Band, EOExtension

    from geosave.geodata import stac
    from geosave.geodata.io.vector import stac_geoparquet as table

    optical = sample["s2"].dataset[["red"]]
    assert optical.red.gs.spectral is None
    files = optical.gs.to_cog(f"{tmp_path}/optical_{{time:%Y%m%d}}_{{band}}.tif")
    rows = table.to_items(stac.create_items(files, collection="s2"))
    # The host states facts its files do not carry, as Planetary Computer does.
    for item in rows:
        EOExtension.ext(item.assets["red"]).bands = [
            Band.create(name="B04", common_name="red", center_wavelength=0.665)
        ]
    combined = pd.concat(
        [items[items["collection"] == "label"], table.from_items(rows)],
        ignore_index=True,
    )

    recovered = cuts.stacks(combined, on="label").iloc[0].gs.to_stack()["s2"].dataset

    assert recovered.red.gs.spectral.common_name == "red"
