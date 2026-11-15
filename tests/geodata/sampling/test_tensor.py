"""The tensor pair must chip and merge exactly as the raster pair a release uses."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from torch.utils.data import default_collate

from geosave.geodata.sampling import (
    RasterChipper,
    RasterMerger,
    TensorChipper,
    TensorMerger,
)

GRID = GeoBox.from_bbox(
    (500_000, 9_000_000, 500_700, 9_000_500),
    crs="EPSG:32749",
    shape=(50, 70),
    tight=True,
)
TIMES = pd.DatetimeIndex(["2025-06-03", "2025-07-08"])


def build_tree(seed: int) -> xr.DataTree:
    """Build a dated two-band scene of random reflectance."""
    pixels = np.random.default_rng(seed).random((2, 2, 50, 70), dtype="float32")
    coords = dict(xr_coords(GRID, always_yx=True))
    coords["time"] = ("time", TIMES)
    dims = ("time", "y", "x")
    scene = xr.Dataset(
        {"b0": (dims, pixels[:, 0]), "b1": (dims, pixels[:, 1])}, coords=coords
    )
    return xr.DataTree.from_dict({"image": scene})


@pytest.mark.parametrize("mode", ["reflect", "edge", "wrap", "constant", "shift"])
def test_tensor_chips_equal_raster_chips(mode):
    trees = [build_tree(0), build_tree(1)]
    batch = default_collate([trees[0].gs.to_tensor(), trees[1].gs.to_tensor()])
    raster = RasterChipper(trees, size=16, stride=12, mode=mode)

    tensor = TensorChipper(batch, pixels=["image"], size=16, stride=12, mode=mode)

    assert len(tensor) == len(raster)
    for index in range(len(raster)):
        expected = raster[index].gs.to_tensor()
        actual = tensor[index]
        assert set(actual) == set(expected)
        torch.testing.assert_close(actual["image"], expected["image"])
        torch.testing.assert_close(actual["transform"], expected["transform"])
        torch.testing.assert_close(actual["time"], expected["time"])
        assert actual["crs"] == expected["crs"]


@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_tensor_merge_equals_raster_merge(window):
    trees = [build_tree(0), build_tree(1)]
    batch = default_collate([trees[0].gs.to_tensor(), trees[1].gs.to_tensor()])
    chipper = TensorChipper(batch, pixels=["image"], size=16, stride=12)
    raster = RasterMerger(chipper.layout, window=window)
    tensor = TensorMerger(chipper.layout, window=window)

    for index in range(len(chipper)):
        chip = chipper[index]["image"][0]
        raster.add(index, chip.numpy())
        tensor.add(index, chip)
    merged = tensor.merge()

    assert merged.shape == (2, 2, 50, 70)
    for raster_index in range(2):
        expected = raster.merge(raster_index).values
        np.testing.assert_allclose(merged[raster_index].numpy(), expected, atol=1e-6)
    # Blending the scene's own chips gives the scene back.
    torch.testing.assert_close(merged, batch["image"][:, 0], atol=1e-5, rtol=0)


def test_chip_samples_keep_the_batch_structure():
    tree = build_tree(0)
    sample = tree.gs.to_tensor()
    scene = {
        "inputs": {"image": sample["image"], "season": torch.tensor([3.0])},
        "target": torch.zeros(50, 70, dtype=torch.long),
        "transform": sample["transform"],
        "crs": sample["crs"],
    }
    batch = default_collate([scene, scene])

    chipper = TensorChipper(batch, pixels=["inputs/image"], size=16, stride=16)
    chips = default_collate([chipper[0], chipper[len(chipper) - 1]])

    assert chips["inputs"]["image"].shape == (2, 2, 2, 16, 16)
    torch.testing.assert_close(chips["inputs"]["season"], torch.tensor([[3.0], [3.0]]))
    # Entries not named in `pixels` are the chip's sample's, uncut.
    assert chips["target"].shape == (2, 50, 70)
    assert chips["crs"] == ["EPSG:32749", "EPSG:32749"]


def test_dropped_edges_stay_unpredicted():
    batch = default_collate([build_tree(0).gs.to_tensor()])
    chipper = TensorChipper(batch, pixels=["image"], size=16, mode="drop")
    merger = TensorMerger(chipper.layout)

    for index in range(len(chipper)):
        merger.add(index, chipper[index]["image"][0])
    merged = merger.merge()

    assert not merged[..., :48, :64].isnan().any()
    assert merged[..., 48:, :].isnan().all()
    assert merged[..., :, 64:].isnan().all()
