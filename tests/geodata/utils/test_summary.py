import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import delayed

import geosave as gs


@pytest.mark.parametrize("as_array", [False, True])
def test_statistics_compute_shared_band_sources_once(as_array):
    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[0.0, 1.0, 3.0], [0.0, 10.0, 20.0]])

    values = da.from_delayed(pixels(), shape=(2, 3), dtype="float64")
    source = xr.Dataset(
        {
            "red": xr.DataArray(values[0], dims="x", attrs={"nodata": 0}),
            # A stacked array keeps the fill every band shares, so both share one.
            "nir": xr.DataArray(values[1], dims="x", attrs={"nodata": 0}),
        }
    )
    if as_array:
        # Conversion requires a spatial pair; a singleton y axis adds no pixels.
        source = source.expand_dims(y=[0]).gs.to_array()

    table = source.gs.statistics()

    assert reads == [1]
    assert list(table.index) == ["red", "nir"]
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["nir", "mean"] == 15.0
    assert table.loc["nir", "valid_percent"] == pytest.approx(200 / 3)


def test_empty_dataset_statistics_keep_the_table_columns():
    table = xr.Dataset().gs.statistics()

    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert list(table.columns) == [
        "minimum",
        "maximum",
        "mean",
        "stddev",
        "valid_percent",
    ]


def test_statistics_pool_the_pixels_of_rasters_of_different_sizes():
    small = xr.Dataset({"red": ("x", [1.0, 3.0])})
    large = xr.Dataset({"red": (("y", "x"), [[5.0, 7.0, 9.0], [11.0, 13.0, 15.0]])})

    table = gs.statistics([small, large])

    pooled = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    assert table.loc["red", "minimum"] == 1.0
    assert table.loc["red", "maximum"] == 15.0
    assert table.loc["red", "mean"] == pytest.approx(pooled.mean())
    assert table.loc["red", "stddev"] == pytest.approx(pooled.std())
    assert table.loc["red", "valid_percent"] == 100.0


def test_statistics_exclude_each_rasters_own_fill():
    first = xr.Dataset({"red": ("x", [0.0, 2.0], {"nodata": 0})})
    second = xr.Dataset({"red": ("x", [9.0, 4.0, np.nan], {"nodata": 9})})

    table = gs.statistics([first, second])

    assert table.loc["red", "mean"] == 3.0
    assert table.loc["red", "valid_percent"] == pytest.approx(40.0)


def test_statistics_read_every_raster_in_one_compute():
    reads = []

    @delayed
    def pixels(value):
        reads.append(value)
        return np.full(4, value)

    rasters = []
    for value in [1.0, 2.0, 3.0]:
        band = da.from_delayed(pixels(value), shape=(4,), dtype="float64")
        rasters.append(xr.Dataset({"red": ("x", band), "nir": ("x", band * 2)}))

    table = gs.statistics(rasters)

    assert sorted(reads) == [1.0, 2.0, 3.0]
    assert list(table.index) == ["red", "nir"]
    assert table.loc["nir", "mean"] == 4.0


def test_statistics_reject_rasters_with_other_variables():
    red = xr.Dataset({"red": ("x", [1.0])})
    nir = xr.Dataset({"nir": ("x", [1.0])})

    with pytest.raises(ValueError, match="nir"):
        gs.statistics([red, nir])


def test_statistics_of_no_rasters_keep_the_table_columns():
    table = gs.statistics([])

    assert table.empty
    assert list(table.columns) == [
        "minimum",
        "maximum",
        "mean",
        "stddev",
        "valid_percent",
    ]


def test_statistics_release_each_rasters_source_before_reading_on():
    import weakref

    import dask

    class Source(np.ndarray):
        """Pixels a file read returned, observed while dask holds them."""

    alive = set()
    most = []

    @delayed
    def read(index):
        pixels = np.full(16, float(index + 1)).view(Source)
        alive.add(index)
        weakref.finalize(pixels, alive.discard, index)
        most.append(len(alive))
        return pixels

    rasters = []
    for index in range(20):
        red = da.from_delayed(read(index), shape=(16,), dtype="float64")
        # Both variables read one source, as NDVI reads the red band.
        rasters.append(xr.Dataset({"red": ("x", red), "double": ("x", red * 2)}))

    with dask.config.set(scheduler="synchronous"):
        gs.statistics(rasters)

    assert max(most) <= 2


def _labels(values):
    band = xr.DataArray(np.array(values, dtype="uint8"), dims=("y", "x"))
    band = band.gs.write_legend({0: "water", 1: "trees", 2: "crops"})
    return band.gs.write_nodata(255)


def test_class_percent_pool_tiles_and_exclude_nodata():
    first = _labels([[0, 0], [1, 255]])
    second = _labels([[1, 2, 255, 255]])

    percent = gs.class_percent([first, second])

    assert list(percent.index) == ["water", "trees", "crops"]
    assert percent["water"] == pytest.approx(100 * 2 / 5)
    assert percent["trees"] == pytest.approx(100 * 2 / 5)
    assert percent["crops"] == pytest.approx(100 * 1 / 5)


def test_class_percent_keep_listed_classes_that_never_occur():
    percent = _labels([[0, 0]]).gs.class_percent()

    assert percent["crops"] == 0.0
    assert percent.sum() == pytest.approx(100.0)


def test_class_percent_refuse_a_band_without_classes():
    band = xr.DataArray([[1.0, 2.0]], dims=("y", "x"))

    with pytest.raises(ValueError, match="legend"):
        band.gs.class_percent()


def test_dataset_class_percent_has_a_column_per_categorical_variable():
    ds = xr.Dataset(
        {
            "label": _labels([[0, 1], [1, 1]]),
            "red": xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("y", "x")),
        }
    )

    table = ds.gs.class_percent()

    assert list(table.columns) == ["label"]
    assert table.loc["trees", "label"] == 75.0


def test_class_percent_read_every_tile_in_one_compute():
    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([0, 1, 1, 255], dtype="uint8")

    values = da.from_delayed(pixels(), shape=(4,), dtype="uint8")
    band = xr.DataArray(values, dims="x")
    band = band.gs.write_legend({0: "water", 1: "trees"}).gs.write_nodata(255)

    percent = band.gs.class_percent()

    assert reads == [1]
    assert percent["trees"] == pytest.approx(200 / 3)


def test_class_percent_drop_the_class_that_names_the_fill():
    band = xr.DataArray(np.array([[0, 1, 255]], dtype="uint8"), dims=("y", "x"))
    band = band.gs.write_legend({0: "water", 1: "trees", 255: "unlabelled"})

    percent = band.gs.write_nodata(255).gs.class_percent()

    assert list(percent.index) == ["water", "trees"]
    assert percent.sum() == pytest.approx(100.0)
