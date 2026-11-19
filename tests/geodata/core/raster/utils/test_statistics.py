import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import delayed

import geosave as gs
from tests.conftest import build_dataset
from tests.geodata.conftest import geobox


def test_statistics_compute_shared_band_sources_once():
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

    table = gs.summary(source)

    assert reads == [1]
    assert list(table.index) == ["red", "nir"]
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["nir", "mean"] == 15.0
    assert table.loc["nir", "valid_percent"] == pytest.approx(200 / 3)


def test_statistics_pool_the_pixels_of_rasters_of_different_sizes():
    small = xr.Dataset({"red": ("x", [1.0, 3.0])})
    large = xr.Dataset({"red": (("y", "x"), [[5.0, 7.0, 9.0], [11.0, 13.0, 15.0]])})

    table = gs.summary([small, large])

    pooled = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    assert table.loc["red", "minimum"] == 1.0
    assert table.loc["red", "maximum"] == 15.0
    assert table.loc["red", "mean"] == pytest.approx(pooled.mean())
    assert table.loc["red", "stddev"] == pytest.approx(pooled.std())
    assert table.loc["red", "valid_percent"] == 100.0


def test_statistics_exclude_each_rasters_own_fill():
    first = xr.Dataset({"red": ("x", [0.0, 2.0], {"nodata": 0})})
    second = xr.Dataset({"red": ("x", [9.0, 4.0, np.nan], {"nodata": 9})})

    table = gs.summary([first, second])

    assert table.loc["red", "mean"] == 3.0
    assert table.loc["red", "valid_percent"] == pytest.approx(40.0)


def test_statistics_reject_rasters_with_other_variables():
    red = xr.Dataset({"red": ("x", [1.0])})
    nir = xr.Dataset({"nir": ("x", [1.0])})

    with pytest.raises(ValueError, match="nir"):
        gs.summary([red, nir])


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
        gs.summary(rasters)

    assert max(most) <= 2


def _labels(values):
    band = xr.DataArray(np.array(values, dtype="uint8"), dims=("y", "x"))
    band = band.gs.write_flags({0: "water", 1: "trees", 2: "crops"})
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
    percent = gs.class_percent(_labels([[0, 0]]))

    assert percent["crops"] == 0.0
    assert percent.sum() == pytest.approx(100.0)


def test_class_percent_refuse_a_band_without_classes():
    band = xr.DataArray([[1.0, 2.0]], dims=("y", "x"))

    with pytest.raises(ValueError, match="classes"):
        gs.class_percent(band)


def test_statistics_reads_a_lazy_source_once() -> None:
    from dask import delayed

    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[1.0, 2.0], [3.0, 0.0]])

    values = da.from_delayed(pixels(), shape=(2, 2), dtype="float64")
    band = xr.DataArray(values, dims=("y", "x"), name="red", attrs={"nodata": 0})
    summary = gs.summary(band)

    assert isinstance(summary, pd.DataFrame)
    assert summary.loc["red", "minimum"] == 1.0
    assert summary.loc["red", "maximum"] == 3.0
    assert summary.loc["red", "mean"] == 2.0
    assert summary.loc["red", "stddev"] == pytest.approx(np.sqrt(2 / 3))
    assert summary.loc["red", "valid_percent"] == 75.0
    assert len(reads) == 1


def test_multiband_array_statistics_keep_band_order_and_shared_nodata() -> None:
    source = build_dataset(
        {
            "nir": (("y", "x"), np.array([[0.0, 10.0, 20.0]])),
            "red": (("y", "x"), np.array([[0.0, 1.0, 3.0]])),
        },
        geobox(shape=(1, 3)),
    ).gs.write_nodata(0)

    table = gs.summary(source.gs.to_array().gs.to_dataset("band"))

    pd.testing.assert_frame_equal(table, gs.summary(source))
    assert list(table.index) == ["nir", "red"]
    assert table.loc["nir", "mean"] == 15.0
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["red", "valid_percent"] == pytest.approx(200 / 3)


def test_raster_statistics_tabulate_each_variable_as_its_band_does() -> None:
    box = geobox()
    pixels = np.arange(64, dtype="uint16").reshape(box.shape)
    built = build_dataset(
        {
            "red": (("y", "x"), pixels),
            "nir": (("y", "x"), pixels * 2),
        },
        box,
        nodata=0,
    )

    table = gs.summary(built)

    assert list(table.index) == ["red", "nir"]
    assert list(table.columns) == [
        "minimum",
        "maximum",
        "mean",
        "stddev",
        "valid_percent",
    ]
    pd.testing.assert_frame_equal(table.loc[["nir"]], gs.summary(built.nir))


@pytest.mark.parametrize("name", ["red", None])
def test_statistics_pool_an_iterable_of_arrays_over_all_axes(name):
    first = xr.DataArray([[1.0, 3.0]], dims=("band", "x"), name=name)
    second = xr.DataArray([[5.0], [7.0]], dims=("band", "x"), name=name)

    table = gs.summary(band for band in [first, second])

    assert list(table.index) == [name]
    assert table.iloc[0]["mean"] == 4.0
    assert table.iloc[0]["stddev"] == pytest.approx(np.std([1, 3, 5, 7]))
    pd.testing.assert_frame_equal(gs.summary(first), gs.summary([first]))


def test_statistics_match_variables_by_name_in_first_raster_order():
    first = xr.Dataset({"red": ("x", [1.0]), "nir": ("x", [10.0])})
    second = xr.Dataset({"nir": ("x", [30.0]), "red": ("x", [3.0])})

    table = gs.summary(iter([first, second]))

    assert list(table.index) == ["red", "nir"]
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["nir", "mean"] == 20.0


def test_class_percent_single_band_matches_a_one_band_iterable():
    band = _labels([[0, 1, 255]])
    before = band.copy(deep=True)

    pd.testing.assert_series_equal(
        gs.class_percent(band), gs.class_percent(b for b in [band])
    )
    xr.testing.assert_identical(band, before)
