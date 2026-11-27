"""Time windows, chips, tensors and merged predictions fit together."""

import numpy as np
import pytest

import geosave as gs
from geosave.geodata.sampling import ChipLocation, RasterChipper, RasterMerger
from tests.geodata.sampling.conftest import build_tree


def test_windows_chip_to_batches_and_merge_back_per_window():
    torch = pytest.importorskip("torch")
    tree = build_tree()
    windows = list(gs.iter_samples([tree], "MS", size=2, keep="last"))
    chips = RasterChipper(windows, size=4, overlap=0.25)
    samples = []
    for location, chip in chips:
        pixels, context = chip.gs.to_tensor(context={"time": "time"})
        location_tensor = torch.tensor(
            [location.source.index, location.top, location.left]
        )
        samples.append({**pixels, **context, "location": location_tensor})
    loader = torch.utils.data.DataLoader(samples, batch_size=5, shuffle=True)
    # Shuffled batches mix windows; the merger places each chip on its own.
    merger = RasterMerger(chips, window="hann")

    for batch in loader:
        assert batch["image"].shape[1:] == (2, 2, 4, 4)
        assert batch["label"].shape[1:] == (1, 4, 4)
        assert batch["time"].shape[1:] == (2,)
        # Stand in for a model: the difference of the window's two dates, per band.
        change = batch["image"][:, 1] - batch["image"][:, 0]
        for (index, top, left), chip_change in zip(
            batch["location"].tolist(), change.numpy(), strict=True
        ):
            source = chips.sources[index]
            merger.add(ChipLocation(source, top, left), chip_change)

    for window, merged in zip(windows, merger.merge(), strict=True):
        image = window["image"].to_dataset()
        after = image.isel(time=1).gs.to_numpy()
        before = image.isel(time=0).gs.to_numpy()
        expected = after - before
        np.testing.assert_allclose(merged.values, expected, rtol=1e-5)
