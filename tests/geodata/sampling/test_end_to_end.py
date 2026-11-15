"""Time windows, chips, tensors and merged predictions fit together."""

import numpy as np
import pytest

from geosave.geodata.sampling import RasterMerger, RasterChipper
from tests.geodata.sampling.conftest import build_tree


def test_windows_chip_to_batches_and_merge_back_per_window():
    torch = pytest.importorskip("torch")
    tree = build_tree()
    windows = tree.gs.sample("MS", size=2, keep="last")
    chipper = RasterChipper(windows, size=4, stride=3)
    merger = RasterMerger(chipper.layout, window="hann")

    samples = []
    for index in range(len(chipper)):
        tensors = chipper[index].gs.to_tensor()
        tensors["index"] = index
        samples.append(tensors)
    loader = torch.utils.data.DataLoader(samples, batch_size=5, shuffle=True)

    for batch in loader:
        assert batch["image"].shape[1:] == (2, 2, 4, 4)
        assert batch["label"].shape[1:] == (1, 4, 4)
        assert batch["time"].shape[1:] == (2,)
        # Stand in for a model: the difference of the window's two dates, per band.
        change = batch["image"][:, 1] - batch["image"][:, 0]
        for index, chip_change in zip(batch["index"].tolist(), change.numpy(), strict=True):
            merger.add(index, chip_change)

    for raster_index, window in enumerate(windows):
        image = window["image"].to_dataset()
        expected = image.isel(time=1).gs.to_numpy() - image.isel(time=0).gs.to_numpy()
        merged = merger.merge(raster_index)
        assert merged.odc.geobox == tree.gs.geobox
        np.testing.assert_allclose(merged.values, expected, rtol=1e-5)
