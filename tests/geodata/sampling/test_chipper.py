"""A chip stream places and cuts one lazy window at a time."""

from itertools import islice

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox

import geosave as gs
from geosave.geodata.errors import RasterValidationError
from geosave.geodata.sampling import (
    ChipLocation,
    RasterChipper,
    lay_chip_offsets,
)
from tests.geodata.sampling.conftest import (
    DATES,
    GRID,
    build_scene,
    build_tree,
    format_times,
)

SMALL_GRID = GeoBox.from_bbox(
    (600_000.0, 9_100_000.0, 600_080.0, 9_100_060.0),
    crs="EPSG:32749",
    shape=(6, 8),
    tight=True,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("shift", [0, 4, 6]),
        ("drop", [0, 4]),
        ("reflect", [0, 4, 8]),
        ("constant", [0, 4, 8]),
    ],
)
def test_each_mode_meets_the_edge_its_own_way(mode, expected):
    assert lay_chip_offsets(10, size=4, overlap=0, mode=mode) == expected


def place(location: ChipLocation) -> tuple[int, int, int]:
    """Return a chip's raster index and top-left pixel."""
    return location.source.index, location.top, location.left


def test_offsets_that_fit_exactly_are_the_same_in_every_mode():
    for mode in ("shift", "drop", "edge", "wrap"):
        assert lay_chip_offsets(8, size=4, overlap=0, mode=mode) == [0, 4]
        assert lay_chip_offsets(10, size=4, overlap=1, mode=mode) == [0, 3, 6]


def test_a_fraction_is_rounded_down_to_pixel_overlap():
    grid = GRID[:4, :10]
    scene = build_tree(grid, times=None)["image"].to_dataset()
    for overlap in (1, 0.25, 0.49):
        chips = RasterChipper(scene, size=4, overlap=overlap)
        assert [where.left for where, _ in chips] == [0, 3, 6]
    assert lay_chip_offsets(10, size=4, overlap=0.0, mode="shift") == [0, 4, 6]
    assert lay_chip_offsets(5, size=4, overlap=0.99, mode="drop") == [0, 1]


@pytest.mark.parametrize("overlap", [-1, 4, -0.1, 1.0, 25.0, float("nan")])
def test_invalid_pixel_counts_and_fractions_are_rejected(overlap):
    with pytest.raises(ValueError, match="overlap"):
        RasterChipper([], size=4, overlap=overlap)


def test_an_axis_shorter_than_a_chip_needs_padding():
    with pytest.raises(ValueError, match="fewer than one chip"):
        lay_chip_offsets(3, size=4, overlap=0, mode="shift")
    assert lay_chip_offsets(3, size=4, overlap=0, mode="drop") == []
    assert lay_chip_offsets(3, size=4, overlap=0, mode="reflect") == [0]


def test_chips_are_cut_lazily():
    tree = build_tree(times=None)

    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        chips = RasterChipper(iter([tree, tree]), size=4)
        iterator = iter(chips)
        first, _ = next(iterator)
        location, chip = next(iterator)
    assert tasks == []
    assert [source.index for source in chips.sources] == [0, 1]
    assert place(first) == (0, 0, 0)
    assert place(location) == (0, 0, 4)
    assert chip.gs.geobox == GRID[0:4, 4:8]
    assert chip["image"]["red"].chunks is not None
    for node, variable in [("image", "red"), ("label", "cover")]:
        np.testing.assert_array_equal(
            chip[node][variable], tree[node][variable][0:4, 4:8]
        )


def test_shift_puts_the_last_chip_flush_with_the_corner():
    chips = list(RasterChipper([build_tree(times=None)], size=4, mode="shift"))
    location, chip = chips[-1]
    assert place(location) == (0, 6, 6)
    assert chip.gs.geobox == GRID[6:10, 6:10]


def test_a_padded_chip_reaches_past_the_edge_and_is_filled():
    tree = build_tree(times=None)
    chips = list(RasterChipper([tree], size=4, mode="constant", constant_value=7))
    location, chip = chips[-1]
    assert len(chips) == 9
    assert place(location) == (0, 8, 8)
    assert chip.gs.geobox.extent.boundingbox.right == GRID.extent.boundingbox.right + 20
    red = chip["image"]["red"].values
    np.testing.assert_array_equal(red[:2, :2], tree["image"]["red"].values[8:, 8:])
    assert (red[2:, :] == 7).all()
    assert (red[:, 2:] == 7).all()
    assert chip["label"]["cover"].dtype == "uint8"


def test_drop_leaves_incomplete_chips_out():
    chips = list(RasterChipper([build_tree(times=None)], size=4, mode="drop"))
    assert len(chips) == 4
    assert chips[-1][1].gs.geobox == GRID[4:8, 4:8]


def test_chips_number_through_several_rasters_on_their_own_grids():
    rasters = [build_tree(), build_tree(SMALL_GRID)]
    chips = list(RasterChipper(rasters, size=4, mode="shift"))
    assert len(chips) == 13
    assert place(chips[0][0]) == (0, 0, 0)
    assert place(chips[8][0]) == (0, 6, 6)
    assert place(chips[9][0]) == (1, 0, 0)
    assert place(chips[12][0]) == (1, 2, 4)
    assert chips[12][1].gs.geobox == SMALL_GRID[2:6, 4:8]


def test_time_windows_are_rasters_like_any_other():
    windows = gs.iter_samples([build_tree()], "MS", size=2, keep="last")
    chips = list(RasterChipper(windows, size=5))
    assert len(chips) == 2 * 4
    assert format_times(chips[0][1]["image"]) == ["06-28", "07-23"]
    assert format_times(chips[4][1]["image"]) == ["09-11", "10-21"]
    assert dict(chips[4][1]["label"].sizes) == {"y": 5, "x": 5}


def test_each_chip_names_its_window_and_the_window_dates():
    windows = list(gs.iter_samples([build_tree()], "MS", size=2, keep="last"))
    sources = []
    for location, _ in RasterChipper(windows, size=5):
        if location.source not in sources:
            sources.append(location.source)

    assert [source.index for source in sources] == list(range(len(windows)))
    assert [source.timespan for source in sources] == [
        window.gs.timespan for window in windows
    ]
    assert sources[0].timespan[0].strftime("%m-%d") == "06-28"
    assert sources[0].geobox == windows[0].gs.geobox

    location, _ = next(iter(RasterChipper(build_tree(times=None), size=5)))
    assert location.source.timespan is None


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
def test_single_rasters_and_streams_yield_identical_windows_and_chips(kind):
    tree = build_tree()
    scene = tree["image"].to_dataset()
    raster = {"array": scene.red, "dataset": scene, "tree": tree}[kind]
    options = dict(size=4, overlap=0, mode="constant", constant_value=17)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        window = next(gs.iter_samples(raster, "MS", size=2))
        location, chip = next(islice(RasterChipper(window, **options), 8, None))
        streamed = gs.iter_samples(iter([raster]), "MS", size=2)
        other_location, other_chip = next(
            islice(RasterChipper(streamed, **options), 8, None)
        )

    assert tasks == []
    assert type(chip) is type(window) is type(raster)
    assert location == other_location
    assert place(location) == (0, 8, 8)
    xr.testing.assert_identical(chip, other_chip)
    if isinstance(chip, xr.DataTree):
        band = chip["image/red"]
    elif isinstance(chip, xr.Dataset):
        band = chip.red
    else:
        band = chip
    assert band.gs.geobox == location.geobox(4)
    expected = scene.red.isel(time=[2, 4], y=slice(8, 10), x=slice(8, 10))
    np.testing.assert_array_equal(band[:, :2, :2], expected)
    assert (band[:, 2:, :] == 17).all()
    assert (band[:, :, 2:] == 17).all()


def test_a_stream_can_mix_raster_types_and_grids():
    tree = build_tree(times=DATES[:1])
    scene = build_tree(SMALL_GRID, times=DATES[-1:])["image"].to_dataset()
    rasters = (tree, scene.red, scene)
    windows = gs.iter_samples(iter(rasters), "D")
    chips = RasterChipper(windows, size=10)
    for index, (source, (location, chip)) in enumerate(
        zip(rasters, chips, strict=True)
    ):
        assert place(location) == (index, 0, 0)
        assert type(chip) is type(source)


def test_a_chip_converts_to_tensors():
    torch = pytest.importorskip("torch")
    windows = gs.iter_samples([build_tree()], "MS", size=2)
    chips = iter(RasterChipper(windows, size=4, overlap=2, mode="reflect"))
    samples = []
    for _ in range(2):
        _, chip = next(chips)
        pixels, context = chip.gs.to_tensor(context={"time": "time"})
        assert pixels["image"].shape == (2, 2, 4, 4)
        assert pixels["label"].shape == (1, 4, 4)
        assert context["time"].shape == (2,)
        samples.append(pixels)
    assert torch.utils.data.default_collate(samples)["image"].shape == (2, 2, 2, 4, 4)


@pytest.mark.parametrize(
    ("mode", "position", "expected_blocks"),
    [
        ("reflect", 0, [(0, 0)]),
        ("constant", 8, [(2, 2)]),
        ("reflect", 8, [(1, 1), (1, 2), (2, 1), (2, 2)]),
    ],
)
def test_tensor_conversion_reads_only_the_source_blocks_needed_by_the_chip(
    mode, position, expected_blocks
):
    pytest.importorskip("torch")
    reads = []

    def record(block, block_id=None):
        reads.append(block_id)
        return block

    source = build_tree(times=None)["image/red"]
    blocks = da.zeros(source.shape, chunks=(4, 4), dtype=source.dtype)
    pixels = blocks.map_blocks(record, meta=np.empty((0, 0), dtype=source.dtype))
    with dask.config.set(scheduler="synchronous"):
        chips = RasterChipper(source.copy(data=pixels), size=4, mode=mode)
        _, chip = next(islice(chips, position, None))
        assert reads == []
        tensor = chip.gs.to_tensor()
    assert tensor.shape == (4, 4)
    assert sorted(reads) == expected_blocks


def test_chips_refuse_incompatible_settings_and_grids():
    tree = build_tree(times=None)
    with pytest.raises(ValueError, match="fewer than one chip"):
        next(iter(RasterChipper([tree], size=16, mode="shift")))
    assert len(list(RasterChipper([tree], size=16))) == 1
    with pytest.raises(ValueError, match="mode must be"):
        next(iter(RasterChipper([tree], size=4, mode="mirror")))
    with pytest.raises(ValueError, match="overlap"):
        RasterChipper([tree], size=4, overlap=4)
    assert list(RasterChipper([], size=4)) == []
    coarse = tree["label"].to_dataset().coarsen(x=2, y=2).max()
    two_grids = xr.DataTree.from_dict(
        {"image": tree["image"].to_dataset(), "coarse": coarse}
    )
    with pytest.raises(RasterValidationError, match="GeoBoxes"):
        next(iter(RasterChipper([two_grids], size=4)))


def test_direct_arrays_and_datasets_need_a_georeferenced_grid():
    band = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), name="red")
    for raster in (band, band.to_dataset()):
        with pytest.raises(RasterValidationError, match="no grid"):
            next(iter(RasterChipper(raster, size=2)))


def test_reflect_is_the_default_and_mirrors_the_raster_edge():
    tree = build_tree(times=None)
    location, chip = list(RasterChipper([tree], size=4))[-1]
    assert place(location) == (0, 8, 8)
    padded = np.pad(tree["image"]["red"].values, ((0, 2), (0, 2)), mode="reflect")
    np.testing.assert_array_equal(chip["image"]["red"], padded[8:12, 8:12])


def test_a_raster_without_chips_keeps_its_position_in_the_input():
    grids = [GRID, GRID[:1, :1], SMALL_GRID]
    rasters = (build_tree(grid, times=None) for grid in grids)

    chips = list(RasterChipper(rasters, size=4, overlap=2, mode="drop"))

    indexes = sorted({location.source.index for location, _ in chips})
    assert indexes == [0, 2]
    for location, chip in chips:
        assert location.geobox(4) == chip.gs.geobox


@pytest.mark.parametrize("chunk_height", [None, 3])
@pytest.mark.parametrize(
    "mode", ["reflect", "constant", "edge", "wrap", "shift", "drop"]
)
def test_batched_chips_equal_the_lazy_chips_loaded(mode, chunk_height):
    torch = pytest.importorskip("torch")
    rasters = [build_tree(), build_tree(SMALL_GRID), build_scene(SMALL_GRID)]
    # Chip rows of 4 with overlap 1 start every 3 rows and straddle 3-row chunks.
    if chunk_height is not None:
        rasters = [raster.chunk(y=chunk_height) for raster in rasters]

    readers = {"transform": "transform", "crs": "crs", "doy": "day_of_year"}
    lazy = []
    for location, chip in RasterChipper(rasters, size=4, overlap=1, mode=mode):
        lazy.append((location, *chip.gs.to_tensor(context=readers)))
    chipper = RasterChipper(rasters, size=4, overlap=1, mode=mode)
    loaded = []
    for batch in chipper.iter_batches(1):
        for location, chip in zip(batch.locations, batch.chips, strict=True):
            loaded.append((location, *chip.gs.to_tensor(context=readers)))

    assert [entry[0] for entry in loaded] == [entry[0] for entry in lazy]
    for (_, pixels, context), (_, expected_pixels, expected_context) in zip(
        loaded, lazy, strict=True
    ):
        if isinstance(expected_pixels, dict):
            assert pixels.keys() == expected_pixels.keys()
            for name in expected_pixels:
                torch.testing.assert_close(
                    pixels[name], expected_pixels[name], equal_nan=True
                )
        else:
            torch.testing.assert_close(pixels, expected_pixels, equal_nan=True)
        assert context.keys() == expected_context.keys()
        for name in expected_context:
            torch.testing.assert_close(context[name], expected_context[name])


def test_batches_read_each_chunk_once_for_chips_smaller_than_chunks():
    pytest.importorskip("torch")
    reads = []

    def record(block, block_id=None):
        reads.append(block_id)
        return block

    source = build_tree(times=None)["image/red"]
    blocks = da.zeros(source.shape, chunks=(4, 4), dtype=source.dtype)
    pixels = blocks.map_blocks(record, meta=np.empty((0, 0), dtype=source.dtype))
    # Two chip rows share each 4-row chunk row.
    chipper = RasterChipper(source.copy(data=pixels), size=2, mode="constant")
    with dask.config.set(scheduler="synchronous"):
        batches = list(chipper.iter_batches(5))

    assert sum(len(batch.chips) for batch in batches) == 25
    assert sorted(reads) == sorted(set(reads))
    assert len(reads) == 9


def test_a_batch_stacks_pixels_and_context_in_chip_order():
    torch = pytest.importorskip("torch")
    rasters = [build_scene(SMALL_GRID), build_scene(SMALL_GRID)]
    readers = {"doy": "day_of_year"}
    chips = []
    for location, chip in RasterChipper(rasters, size=4, overlap=1):
        chips.append((location, chip.gs.to_tensor()))

    chipper = RasterChipper(rasters, size=4, overlap=1)
    batches = list(chipper.iter_batches(3))

    sizes = [len(batch.locations) for batch in batches]
    assert sizes[:-1] == [3] * (len(batches) - 1)
    assert sum(sizes) == len(chips)

    locations = []
    for batch in batches:
        locations.extend(batch.locations)
    assert locations == [location for location, _ in chips]

    pixels, context = batches[0].to_tensor(context=readers)
    expected = torch.stack([tensor for _, tensor in chips[:3]])
    torch.testing.assert_close(pixels, expected, equal_nan=True)
    assert list(context) == ["doy"]
    assert context["doy"].shape[0] == 3
    torch.testing.assert_close(batches[0].to_tensor(), expected, equal_nan=True)


def test_tree_batches_key_pixels_by_node_path():
    pytest.importorskip("torch")
    rasters = [build_tree(), build_tree(SMALL_GRID)]

    pixels = next(RasterChipper(rasters, size=4).iter_batches(2)).to_tensor()

    assert list(pixels) == ["image", "label"]
    assert pixels["image"].shape[0] == 2
