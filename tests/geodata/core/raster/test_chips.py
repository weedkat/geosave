"""Tile layout over xarray sources and native Tiler merging."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask import delayed
from dask.callbacks import Callback
from odc.geo.geobox import GeoBox
from tiler import Merger, Tiler

from geosave.geodata import Chips, open_raster
from tests.conftest import build_dataset
from tests.conftest import place


def scene(shape=(5, 7)):
    grid = GeoBox(shape, Affine(10, 2, 123, 1, -10, 456), "EPSG:32633")
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    return build_dataset({"image": (("y", "x"), values)}, grid)


def test_irregular_mode_rejected_before_computing_sources():
    source = scene().chunk()
    with Callback(pretask=lambda *_: pytest.fail("computed rejected source")):
        with pytest.raises(ValueError, match="fixed-size"):
            Chips([source], size=4, mode="irregular")


def test_unreferenced_source_rejected():
    with pytest.raises(ValueError, match="grid"):
        Chips([xr.DataArray(np.zeros((4, 4)), dims=("y", "x"))], size=4)


@pytest.mark.parametrize("mode", ["constant", "reflect", "edge", "wrap", "drop"])
@pytest.mark.parametrize("overlap", [0, 2, 0.5])
@pytest.mark.parametrize("shape", [(5, 7), (1, 2)])
def test_tiles_match_native_tiler(mode, overlap, shape):
    source = scene(shape).chunk({"y": 2})
    chips = Chips([source], size=4, overlap=overlap, mode=mode, constant_value=-1)
    pixels = source.gs.to_numpy()
    tiler = Tiler(
        pixels.shape,
        (1, 4, 4),
        channel_dimension=0,
        overlap=overlap,
        mode=mode,
        constant_value=-1,
    )
    assert len(chips) == len(tiler)
    for key in chips:
        chip = chips[key]
        assert chip.gs.geobox == chips.geobox(key)
        np.testing.assert_array_equal(
            chip.gs.to_numpy(), tiler.get_tile(pixels, key[1])
        )


@pytest.mark.parametrize("as_dataset", [False, True])
def test_construction_is_lazy_and_chip_reads_only_needed_chunks(as_dataset):
    read = []

    @delayed
    def block(value):
        read.append(value)
        return np.full((4, 4), value, dtype="float32")

    rows = []
    for row in ((1, 2), (3, 4)):
        blocks = []
        for value in row:
            blocks.append(da.from_delayed(block(value), (4, 4), "float32"))
        rows.append(da.concatenate(blocks, axis=1))
    grid = GeoBox((8, 8), Affine(10, 0, 0, 0, -10, 80), "EPSG:32633")
    source = build_dataset({"image": (("y", "x"), da.concatenate(rows))}, grid)
    if not as_dataset:
        source = source.image
    with Callback(pretask=lambda *_: pytest.fail("construction computed pixels")):
        chips = Chips([source], size=4)
        chip = chips[0]
    assert read == []
    pixels = chip.gs.to_numpy()
    np.testing.assert_array_equal(pixels, np.ones_like(pixels))
    assert read == [1]


def test_flat_index_and_key_address_the_same_chip():
    source = scene()
    chips = Chips([source, source], size=4)
    keys = list(chips)
    assert len(keys) == len(chips)
    for index, key in enumerate(keys):
        assert chips.key(index) == key
        xr.testing.assert_identical(chips[index], chips[key])
    with pytest.raises(IndexError):
        chips[len(chips)]
    with pytest.raises(IndexError):
        chips[-1]


def test_chip_carries_model_context_from_its_source():
    source = scene((8, 8)).expand_dims(
        time=np.array(["2025-01-05", "2025-07-15"], "M8[ns]")
    )
    source = source.gs.write_nodata(-1).gs.write_extent(source.gs.geobox.extent)
    chips = Chips([source], size=4)
    first, last = chips[0], chips[len(chips) - 1]
    xr.testing.assert_identical(first.time, source.time)
    assert first.image.gs.nodata == -1
    assert (
        first.gs.region.extent.symmetric_difference(chips.geobox(0).extent).area < 1e-6
    )
    assert first.gs.region.extent.centroid != last.gs.region.extent.centroid


def test_temporal_axes_and_multiple_samples_retain_array_order():
    source = scene().assign(other=scene().image * 2)
    source = source.expand_dims(time=[1, 2]).transpose("y", "time", "x")
    samples = [source, source.isel(x=slice(3))]
    chips = Chips(samples, size=4, mode="reflect")
    assert chips[0, 0].gs.to_numpy().shape == (2, 2, 4, 4)
    predictions = []
    for key in reversed(list(chips)):
        predictions.append((key, chips[key].gs.to_numpy()))
    outputs = chips.merge(predictions)
    for sample, actual in zip(samples, outputs, strict=True):
        np.testing.assert_allclose(actual, sample.gs.to_numpy())
    assert chips.xarray[0] is source


@pytest.mark.parametrize("window", [None, "hamming", np.ones((1, 4, 4))])
def test_merge_can_be_repeated_and_keeps_native_spatial_names(window):
    source = scene().rename(y="latitude", x="longitude", spatial_ref="crs")
    source.image.encoding["grid_mapping"] = "crs"
    raster = source.image
    chips = Chips([raster], size=4, overlap=2, mode="reflect")
    assert chips[0].dims == ("latitude", "longitude")
    for _ in range(2):
        predictions = []
        for key in chips:
            predictions.append((key, chips[key].values))
        output = chips.merge(predictions, window=window)[0]
        np.testing.assert_allclose(output, source.image, atol=1e-5)
        assert chips.xarray[0] is raster


def test_merged_prediction_round_trips_on_source_grid(tmp_path):
    source = scene()
    chips = Chips([source], size=4, mode="reflect")
    predictions = []
    for key in chips:
        predictions.append((key, chips[key].gs.to_numpy() * 2))
    (merged,) = chips.merge(predictions)
    output = xr.DataArray(merged[0], dims=("y", "x"), name="prediction")
    output = output.pipe(place, source.gs.geobox)
    path = output.gs.to_cog(tmp_path / "prediction.tif")
    with open_raster(path) as restored:
        np.testing.assert_allclose(restored.prediction, source.image * 2)
        assert restored.gs.geobox == source.gs.geobox


@pytest.mark.parametrize("mode", ["reflect", "drop"])
@pytest.mark.parametrize("window", [None, "hann"])
def test_partial_repeated_predictions_follow_native_merger(mode, window):
    source = scene()
    chips = Chips([source], size=4, overlap=2, mode=mode)
    pixels = source.gs.to_numpy()
    tiler = Tiler(pixels.shape, (1, 4, 4), channel_dimension=0, overlap=2, mode=mode)
    merger = Merger(tiler, window=window)
    predictions = []
    # Omit all other tiles and submit different values for this tile twice.
    for value in (2.0, 6.0):
        tile = np.full((1, 4, 4), value, dtype="float32")
        predictions.append(((0, 0), tile))
        merger.add(0, tile)
    actual = chips.merge(predictions, window=window)[0]
    np.testing.assert_array_equal(actual, merger.merge())


def test_sample_without_predictions_merges_to_none():
    source = scene()
    chips = Chips([source, source.isel(x=slice(3))], size=4)
    predictions = []
    for key in chips:
        if key[0] == 1:
            pixels = chips[key].gs.to_numpy()
            predictions.append((key, np.concatenate([pixels, pixels])))
    outputs = chips.merge(predictions)
    assert outputs[0] is None
    assert outputs[1].shape == (2, 5, 3)
    np.testing.assert_array_equal(outputs[1][0], source.image.values[:, :3])
    assert chips.merge([]) == [None, None]


def test_prediction_leading_shapes_cannot_silently_change_axis_meaning():
    chips = Chips([scene()], size=4)
    predictions = [((0, 0), np.ones((2, 3, 4, 4))), ((0, 1), np.ones((3, 2, 4, 4)))]
    with pytest.raises(ValueError, match="leading shape"):
        chips.merge(predictions)


@pytest.mark.parametrize("overlap", [0, 2])
def test_chip_geobox_preserves_rotated_grid_and_padded_edge(overlap):
    source = scene().chunk()
    chips = Chips([source, source], size=4, overlap=overlap)
    tiler = Tiler((5, 7), (4, 4), overlap=overlap)
    key = (1, len(tiler) - 1)
    (top, left), _ = tiler.get_tile_bbox(key[1])
    expected = GeoBox(
        (4, 4),
        source.gs.geobox.transform * Affine.translation(left, top),
        source.gs.geobox.crs,
    )
    with Callback(pretask=lambda *_: pytest.fail("grid lookup computed pixels")):
        assert chips.geobox(key) == expected
        assert chips.geobox(len(chips) - 1) == expected
