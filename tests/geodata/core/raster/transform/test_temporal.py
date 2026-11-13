from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from geosave.geodata.io import open_netcdf

from tests.conftest import build_dataset as build_raster
from tests.geodata.conftest import geobox

OBSERVED = np.array(["2024-01-05", "2024-01-20", "2024-03-08"], "datetime64[ns]")


def series(
    *values: int, nodata: int | None = None, dtype: str = "uint16"
) -> xr.Dataset:
    """Build a raster whose every pixel reads `values[i]` at `OBSERVED[i]`."""
    box = geobox(shape=(2, 2))
    pixels = np.stack([np.full(box.shape, value, dtype) for value in values])
    return build_raster(
        {"red": (("time", *("y", "x")), pixels)},
        box,
        time=OBSERVED,
        nodata=nodata,
    )


def _cells(raster) -> list[tuple[str, str]]:
    """Read each cell as day strings."""
    starts = raster.start_datetime.values.astype("datetime64[D]").astype(str)
    ends = raster.end_datetime.values.astype("datetime64[D]").astype(str)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _pandas_bins(labels, freq, **options) -> list[tuple[np.datetime64, np.datetime64]]:
    """Read pandas' own resample bins, to check composite cells against."""
    import pandas as pd

    binner, _, named = pd.Grouper(freq=freq, **options)._get_time_bins(
        pd.DatetimeIndex(labels)
    )
    edges = list(zip(binner[:-1], binner[1:], strict=True))[-len(named) :]
    return [(np.datetime64(a, "ns"), np.datetime64(b, "ns")) for a, b in edges]


def test_a_composite_cell_is_its_month_end_bin() -> None:
    monthly = series(100, 900, 500).gs.mask_and_scale().gs.composite("ME", "median")

    # "ME" labels each month by its last day and closes bins on the right.
    assert monthly.time.values.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-31",
        "2024-02-29",
        "2024-03-31",
    ]
    assert _cells(monthly) == [
        ("2023-12-31", "2024-01-31"),
        ("2024-01-31", "2024-02-29"),
        ("2024-02-29", "2024-03-31"),
    ]


def test_a_composite_of_composites_states_the_outer_bin() -> None:
    quarterly = (
        series(100, 900, 500)
        .gs.mask_and_scale()
        .gs.composite("MS", "max")
        .gs.mask_and_scale()
        .gs.composite("QS", "max")
    )

    assert _cells(quarterly) == [("2024-01-01", "2024-04-01")]


@pytest.mark.parametrize(
    "freq, options",
    [
        ("MS", {}),
        ("W", {}),
        ("10D", {"closed": "right", "label": "right"}),
    ],
)
def test_a_composite_cell_is_its_resample_bin(freq, options) -> None:
    data = series(100, 900, 500).gs.mask_and_scale()

    result = data.gs.composite(freq, "mean", **options)

    expected = _pandas_bins(OBSERVED, freq, **options)
    actual = list(
        zip(result.start_datetime.values, result.end_datetime.values, strict=True)
    )
    assert actual == expected


def _scenes_with_properties() -> xr.Dataset:
    properties = [
        {"platform": "s2a", "eo:cloud_cover": 10.0, "constellation": "s2"},
        {"platform": "s2b", "eo:cloud_cover": 20.0, "constellation": "s2"},
        {"platform": "s2a", "constellation": "s2"},
    ]
    return series(100, 900, 500).assign_coords(stac_properties=("time", properties))


def test_composite_properties_keep_what_every_scene_states() -> None:
    monthly = _scenes_with_properties().gs.mask_and_scale().gs.composite("MS", "median")

    assert monthly.stac_properties.values[0] == {"constellation": "s2"}
    # February observed nothing, so it states nothing.
    assert monthly.stac_properties.values[1] == {}


def test_composite_properties_stack_one_entry_per_scene() -> None:
    monthly = (
        _scenes_with_properties()
        .gs.mask_and_scale()
        .gs.composite("MS", "median", properties="stack")
    )

    january = monthly.stac_properties.values[0]
    assert january == {
        "platform": ["s2a", "s2b"],
        "eo:cloud_cover": [10.0, 20.0],
        "constellation": ["s2", "s2"],
    }


def test_flag_values_refuse_a_reducer_that_ranks_them() -> None:
    # min returns a real class code, which is why it reads as safe and is not:
    # it asserts an order over codes that carry none.
    coded = series(4, 5, 4, dtype="uint8").gs.write_flags(
        {4: "veg", 5: "water"}, target=["red"]
    )

    with pytest.raises(ValueError, match="class codes"):
        coded.gs.mask_and_scale().gs.composite("MS", "min")


def test_flag_values_take_a_point_sample() -> None:
    coded = series(4, 5, 4, dtype="uint8").gs.write_flags(
        {4: "veg", 5: "water"}, target=["red"]
    )

    sampled = coded.gs.mask_and_scale().gs.composite("MS", "first")

    assert sampled.red.values[0, 0, 0] == 4
    assert sampled.red.attrs["cell_methods"] == "time: point"


def test_the_nodata_value_is_left_out_of_the_reduction() -> None:
    # January reads its nodata value and 900, so the month observed only 900.
    scene = series(0, 900, 500, nodata=0)

    assert (
        scene.gs.mask_and_scale().gs.composite("MS", "mean").red.values[0, 0, 0] == 900
    )


def monthly(*values: int) -> xr.Dataset:
    """Collapse `series` into monthly buckets, February observing none."""
    return series(*values).gs.mask_and_scale().gs.composite("MS", "median")


def test_a_bucket_observing_nothing_keeps_its_cell_and_absence() -> None:
    buckets = monthly(100, 100, 900)

    # resample cuts every bucket its cadence covers, so nothing is missing from
    # the axis; February keeps its cell and has no pixels.
    assert [str(label)[:7] for label in buckets.time.values] == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]
    assert _cells(buckets)[1] == ("2024-02-01", "2024-03-01")
    assert bool(buckets.red.isel(time=1).isnull().all())


def test_composite_requires_explicit_decode_and_reduces_physical_values():
    from dask.callbacks import Callback

    source = series(1, 3, 5).chunk(time=1)
    source.red.attrs.update(
        scale_factor=2.0,
        add_offset=10.0,
        units="m",
        standard_name="height",
        cell_methods="area: mean",
    )
    data = source
    with Callback(pretask=lambda *_: pytest.fail("read pixels during validation")):
        with pytest.raises(ValueError, match="mask_and_scale"):
            data.gs.composite("MS", "sum")

    decoded = data.gs.mask_and_scale()
    for method, expected in (("sum", 28.0), ("std", 2.0), ("var", 4.0)):
        with Callback(pretask=lambda *_: pytest.fail("read pixels during reduction")):
            result = decoded.gs.composite("MS", method)
        variable = result.red
        assert variable.isel(time=0).compute().values[0, 0] == expected
        assert variable.encoding == {}
        assert variable.attrs["cell_methods"].startswith("area: mean time: ")
        assert variable.attrs["standard_name"] == "height"
        if method == "var":
            assert "units" not in variable.attrs
        else:
            assert variable.attrs["units"] == "m"


def test_composite_band_roundtrips_observation_coordinates(tmp_path):
    source = _scenes_with_properties().gs.composite("MS", "mean")
    band = source.red
    path = tmp_path / "monthly.nc"
    band.gs.to_netcdf(path)
    with open_netcdf(path) as stored:
        restored = stored.red
        xr.testing.assert_allclose(restored, band)
        assert (
            restored.stac_properties.values.tolist()
            == band.stac_properties.values.tolist()
        )
        assert restored.gs.timespan == band.gs.timespan
        assert restored.gs.crs == band.gs.crs


def test_composite_interpolates_pixel_gaps_lazily():
    from dask.callbacks import Callback

    data = series(1, 3, 9, dtype="float32")
    data.red.values[0, 0, 0] = np.nan
    data.red.values[1, 0, 0] = np.nan
    data = data.chunk({"time": 1})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = data.gs.composite("MS", "mean", interpolation="linear")
    assert not tasks
    native = data.resample(time="MS").mean(keep_attrs=True)
    expected = native.chunk(time=-1).interpolate_na(dim="time", method="linear")
    actual = result.red
    np.testing.assert_allclose(actual.compute(), expected.red.compute(), equal_nan=True)
    assert np.isnan(actual.isel(time=0, y=0, x=0).compute())
    assert np.isfinite(actual.isel(time=1, y=1, x=1).compute())
    assert result.start_datetime.values[1] == np.datetime64("2024-02-01", "ns")
    assert actual.attrs["cell_methods"] == "time: mean"


def test_composite_refuses_interpolation_of_class_codes():
    data = series(1, 3, 9).gs.write_flags(
        {1: "water", 3: "trees", 9: "cloud"}, target=["red"]
    )
    with pytest.raises(ValueError, match="class codes"):
        data.gs.composite("MS", "first", interpolation="linear")


def test_input_cells_are_replaced_not_reduced():
    data = series(1, 3, 9, dtype="float32")
    data = data.assign_coords(
        start_datetime=("time", OBSERVED - np.timedelta64(1, "D")),
        end_datetime=("time", OBSERVED + np.timedelta64(1, "D")),
    )
    result = data.gs.composite("MS", "mean")

    assert _cells(result) == [
        ("2024-01-01", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-03-01", "2024-04-01"),
    ]
