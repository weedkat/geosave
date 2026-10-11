from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata import GeoVector, cuts, stac
from geosave_engine.geodata.core.stack import stack
from geosave_engine.ml.inputs import model_inputs
from torch.utils.data import Dataset
from tiler import Merger

from tests.geodata.conftest import stack_templates


_spec_processing_not_settled = pytest.mark.xfail(
    raises=NotImplementedError,
    strict=True,
    reason="ModelSpec processing is a stub until frames, chips, inputs and context are redesigned",
)

torch = pytest.importorskip("torch")

# Undated parents are catalogued as observed on one day.
WHEN = datetime(2025, 1, 1, tzinfo=UTC)


def _raster(height: int, width: int, seed: int = 0) -> xr.Dataset:
    # Each seed lies in its own place: rasters sharing a place and a time are one window.
    west = seed * 10_000
    grid = GeoBox.from_bbox(
        (west, 0, west + width * 10, height * 10), "EPSG:32748", resolution=10
    )
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "B04": (("y", "x"), rng.random((height, width)).astype("float32")),
            "B08": (("y", "x"), rng.random((height, width)).astype("float32")),
        },
        coords=xr_coords(grid),
    )


def origin(windows, frames, position) -> tuple[int, int]:
    """Return a chip's first pixel in its parent's grid, as (row, col)."""
    chip = windows.iloc[position]
    mine, theirs = (
        chip["proj:transform"],
        frames.iloc[int(chip["parent"])]["proj:transform"],
    )
    return (
        round((mine[5] - theirs[5]) / theirs[4]),
        round((mine[2] - theirs[2]) / theirs[0]),
    )


class Samples(Dataset):
    """Consumer-owned Dataset exercising the public read and input APIs."""

    def __init__(self, frames, windows, tilers, padding, *, lone, spec=None):
        self.frames, self.windows = frames, windows
        self.tilers, self.padding = tilers, padding
        self.lone, self.spec = lone, spec
        self.target = "label"

    def __len__(self):
        return len(self.windows)

    def rasters(self, position):
        return self.windows.iloc[position].gs.to_stack().gs.rasters

    def read(self, position):
        rasters = {k: v for k, v in self.rasters(position).items() if k != "label"}
        return rasters["image"] if self.lone else stack(rasters)

    def origin(self, position) -> tuple[int, int]:
        """Return the chip's first pixel in its parent's grid, as (row, col)."""
        return origin(self.windows, self.frames, position)

    def __getitem__(self, position):
        tile = self.read(position)
        inputs = (
            {"image": tile.gs.to_tensor()}
            if self.spec is None
            else model_inputs(self.spec, self.rasters(position))
        )
        return inputs, position


def _samples(parents, shape, root, *, overlap=0, mode="reflect", spec=None):
    """Save each parent as COGs with a label to anchor on, and cut its chips.

    A lone raster is the group `image`; a parent without a `label` group gets
    one, its first band as `uint8`.
    """
    tables, tilers, padding = [], [], []
    lone = not isinstance(next(iter(parents.values())), xr.DataTree)
    for key, parent in parents.items():
        rasters = dict(parent.gs.rasters) if not lone else {"image": parent}
        if "label" not in rasters:
            first = next(iter(rasters.values()))
            band = first[next(iter(first.data_vars))]
            band = band.isel(time=0, drop=True) if "time" in band.dims else band
            rasters["label"] = band.astype("uint8").to_dataset(name="label")
        sample = stack(rasters)
        files = sample.gs.to_cog(stack_templates(sample, f"{root}/{key}"))
        tables.append(stac.create_items(files, id=key, datetime=WHEN))
        tiler, halo = cuts.layout(
            (sample.gs.geobox.shape[0], sample.gs.geobox.shape[1]),
            shape,
            overlap=overlap,
            mode=mode,
        )
        tilers.append(tiler)
        padding.append(halo)
    frames = cuts.stacks(GeoVector.concat(tables), on="label")
    windows = cuts.chips(frames, shape, overlap=overlap, mode=mode)
    return Samples(frames, windows, tilers, padding, lone=lone, spec=spec)


def test_variables_are_read_in_the_order_they_were_written_in(tmp_path) -> None:
    source = _raster(60, 60)

    forward, _ = _samples({"a": source[["B04", "B08"]]}, (32, 32), tmp_path / "f")[0]
    reversed_, _ = _samples({"a": source[["B08", "B04"]]}, (32, 32), tmp_path / "r")[0]

    assert torch.equal(forward["image"][0], reversed_["image"][1])
    assert torch.equal(forward["image"][1], reversed_["image"][0])


def test_a_stack_reads_one_tensor_per_group(tmp_path) -> None:
    grouped = stack({"optical": _raster(60, 60), "dem": _raster(60, 60) * 2})
    samples = _samples({"a": grouped}, (32, 32), tmp_path, overlap=8)

    inputs, index = samples[0]
    image = inputs["image"]
    assert index == 0
    assert sorted(image) == ["dem", "optical"]
    assert image["optical"].shape == (2, 32, 32)


@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_a_small_model_and_shuffled_native_merge_rebuild_each_parent(window, tmp_path):
    from torch.utils.data import DataLoader

    parents = {"a": _raster(13, 19, seed=1), "b": _raster(13, 19, seed=2)}
    samples = _samples(parents, (6, 8), tmp_path, overlap=2)
    mergers = [
        Merger(layout, logits=3, window=window, save_visits=False)
        for layout in samples.tilers
    ]
    model = torch.nn.Conv2d(2, 3, 1).eval()
    with torch.inference_mode():
        for inputs, ids in DataLoader(
            samples,
            batch_size=3,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        ):
            for position, output in zip(
                ids.tolist(), model(inputs["image"]).numpy(), strict=True
            ):
                row = samples.windows.iloc[position]
                mergers[int(row.parent)].add(int(row.chip), output)
        for position, parent in enumerate(parents.values()):
            actual = mergers[position].merge(extra_padding=samples.padding[position])
            expected = model(parent.gs.to_tensor()[None])[0].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_window_positions_associate_outputs_after_a_saved_shuffle(tmp_path) -> None:
    from torch.utils.data import DataLoader

    from geosave_engine.geodata import open_vector

    sources = [_raster(12, 16, seed=1), _raster(12, 16, seed=2)]
    samples = _samples(
        dict(zip(["a", "b"], sources, strict=True)), (6, 8), tmp_path, overlap=2
    )
    saved = open_vector(samples.windows.gs.to_geoparquet(tmp_path / "w.parquet"))
    lookup = saved.sample(frac=1, random_state=41)
    loader = DataLoader(
        samples, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(41)
    )
    observed = set()
    for inputs, ids in loader:
        for position, image in zip(ids.tolist(), inputs["image"], strict=True):
            row = lookup.loc[position]
            row_off, col_off = samples.origin(position)
            parent = sources[int(row.parent)].gs.to_tensor()
            inside = image[:, max(0, -row_off) :, max(0, -col_off) :]
            top, left = max(0, row_off), max(0, col_off)
            expected = parent[
                :, top : top + inside.shape[1], left : left + inside.shape[2]
            ]
            torch.testing.assert_close(
                inside[:, : expected.shape[1], : expected.shape[2]], expected
            )
            observed.add(position)
    assert observed == set(range(len(samples)))


@pytest.mark.parametrize("mode", ["reflect", "edge", "constant"])
def test_a_chip_matches_native_fringe_padding(mode, tmp_path):
    parent = _raster(13, 19)
    samples = _samples({"a": parent}, (6, 8), tmp_path, overlap=2, mode=mode)
    position = len(samples) - 1
    actual, _ = samples[position]
    # Overlapping chips sit over a halo, filled the way the fringe is.
    native, halo = cuts.layout((13, 19), (6, 8), overlap=2, mode=mode)
    fill = {"constant_values": np.nan} if mode == "constant" else {}
    padded = np.pad(parent.gs.to_tensor().numpy(), [(0, 0), *halo], mode=mode, **fill)
    expected = np.stack([native.get_tile(band, position) for band in padded])
    np.testing.assert_array_equal(actual["image"].numpy(), expected)


@_spec_processing_not_settled
def test_named_inputs_preserve_different_leading_axes_and_tile_grid(tmp_path):
    from geosave_engine.model.spec import ModelSpec, Ref

    optical = _raster(13, 19).expand_dims(
        time=np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")
    )
    dem = _raster(13, 19)[["B04"]]
    parent = stack({"optical": optical, "dem": dem})
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {
                "optical": {"variables": ["B04", "B08"]},
                "dem": {"variables": ["B04"]},
            },
            "inputs": {"image": Ref("optical"), "height": Ref("dem")},
        }
    )
    samples = _samples({"a": parent}, (6, 8), tmp_path, overlap=2, spec=spec)
    position = len(samples) - 1
    inputs, _ = samples[position]
    assert inputs["image"].shape == (2, 2, 6, 8)
    assert inputs["height"].shape == (1, 6, 8)
    row_off, col_off = samples.origin(position)
    expected = parent.gs.geobox.translate_pix(col_off, row_off).crop((6, 8))
    assert samples.read(position).gs.geobox == expected


def test_a_constant_halo_is_filled_with_missing_values(tmp_path):
    source = _raster(6, 8)[["B04"]]
    samples = _samples({"a": source}, (4, 4), tmp_path, overlap=2, mode="constant")
    layout, padding = samples.tilers[0], samples.padding[0]
    padded = np.pad(source.B04.values, padding, mode="constant", constant_values=np.nan)
    for index in (0, len(samples) - 1):
        actual = samples.read(index).B04.values
        np.testing.assert_allclose(
            actual, layout.get_tile(padded, index), equal_nan=True
        )
