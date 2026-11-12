from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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


def test_a_left_label_starts_its_cell() -> None:
    # "ME" with label="left" names January's bin by December 31.
    monthly = (
        series(100, 900, 500)
        .gs.mask_and_scale()
        .gs.composite("ME", "median", label="left")
    )

    assert str(monthly.time.values[0])[:10] == "2023-12-31"
    assert _cells(monthly)[0] == ("2023-12-31", "2024-01-31")


def test_a_composite_of_composites_states_the_outer_bin() -> None:
    quarterly = (
        series(100, 900, 500)
        .gs.mask_and_scale()
        .gs.composite("MS", "max")
        .gs.mask_and_scale()
        .gs.composite("QS", "max")
    )

    assert _cells(quarterly) == [("2024-01-01", "2024-04-01")]


@pytest.mark.parametrize("as_array", [False, True])
@pytest.mark.parametrize(
    "freq, options",
    [
        ("MS", {}),
        ("ME", {}),
        ("QE", {}),
        ("W", {}),
        ("10D", {"closed": "right", "label": "right"}),
        ("10D", {"offset": "2D"}),
        ("7D", {"origin": "end"}),
        ("36h", {"offset": "6h"}),
    ],
)
def test_a_composite_cell_is_its_resample_bin(freq, options, as_array) -> None:
    data = series(100, 900, 500).gs.mask_and_scale()
    data = data.red if as_array else data

    result = data.gs.composite(freq, "mean", **options)

    expected = _pandas_bins(OBSERVED, freq, **options)
    actual = list(
        zip(result.start_datetime.values, result.end_datetime.values, strict=True)
    )
    assert actual == expected


def test_one_variable_keeps_its_period() -> None:
    monthly = series(100, 900, 500).gs.mask_and_scale().gs.composite("MS", "median")

    assert "start_datetime" in monthly.red.coords
    assert monthly.isel(time=0).red.gs.timespan is not None


def _scenes_with_properties() -> xr.Dataset:
    from geosave.geodata.stac.raster import stamp_properties_coordinate

    return stamp_properties_coordinate(
        series(100, 900, 500),
        [
            {"platform": "s2a", "eo:cloud_cover": 10.0, "constellation": "s2"},
            {"platform": "s2b", "eo:cloud_cover": 20.0, "constellation": "s2"},
            {"platform": "s2a", "constellation": "s2"},
        ],
    )


def test_composite_properties_keep_what_every_scene_states() -> None:
    import json

    monthly = _scenes_with_properties().gs.mask_and_scale().gs.composite("MS", "median")

    assert json.loads(str(monthly.stac_properties.values[0])) == {"constellation": "s2"}
    # February observed nothing, so it states nothing.
    assert monthly.stac_properties.values[1] == "{}"


def test_composite_properties_stack_one_entry_per_scene() -> None:
    import json

    monthly = (
        _scenes_with_properties()
        .gs.mask_and_scale()
        .gs.composite("MS", "median", properties="stack")
    )

    january = json.loads(str(monthly.stac_properties.values[0]))
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


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_reduction_preserves_metadata_under_xarray_drop_attrs_option(kind):
    from dask.callbacks import Callback

    source = series(100, 900, 500).chunk({"time": 1})
    source.red.attrs["units"] = "m"
    data = source.red if kind == "array" else source
    with (
        xr.set_options(keep_attrs=False),
        Callback(pretask=lambda *_: pytest.fail("reduction computed lazy pixels")),
    ):
        result = data.gs.mask_and_scale().gs.composite("MS", "mean")

    variable = result if kind == "array" else result.red
    assert variable.attrs["units"] == "m"
    assert variable.attrs["cell_methods"] == "time: mean"
    assert "start_datetime" in result.coords
    assert variable.chunks is not None
    np.testing.assert_allclose(variable.compute()[:, 0, 0], [500, np.nan, 500])


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_composite_requires_explicit_decode_and_reduces_physical_values(kind):
    from dask.callbacks import Callback

    source = series(1, 3, 5).chunk(time=1)
    source.red.attrs.update(
        scale_factor=2.0,
        add_offset=10.0,
        units="m",
        standard_name="height",
        cell_methods="area: mean",
    )
    data = source.red if kind == "array" else source
    with Callback(pretask=lambda *_: pytest.fail("read pixels during validation")):
        with pytest.raises(ValueError, match="mask_and_scale"):
            data.gs.composite("MS", "sum")

    decoded = data.gs.mask_and_scale()
    for method, expected in (("sum", 28.0), ("std", 2.0), ("var", 4.0)):
        with Callback(pretask=lambda *_: pytest.fail("read pixels during reduction")):
            result = decoded.gs.composite("MS", method)
        variable = result if kind == "array" else result.red
        assert variable.isel(time=0).compute().values[0, 0] == expected
        assert variable.encoding == {}
        assert variable.attrs["cell_methods"].startswith("area: mean time: ")
        assert variable.attrs["standard_name"] == "height"
        if method == "var":
            assert "units" not in variable.attrs
        else:
            assert variable.attrs["units"] == "m"


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_an_empty_bin_keeps_its_cell_in_selection_and_timespan(kind):
    data = series(1, 3, 5)
    if kind == "array":
        data = data.red
    monthly = data.gs.composite("MS", "mean")

    # Cells are half-open like resample bins: each ends where the next starts.
    assert monthly.sel(time="2024-02").sizes["time"] == 1
    assert monthly.end_datetime.values[0] == monthly.start_datetime.values[1]
    start, end = monthly.gs.timespan
    assert start.isoformat() == "2024-01-01T00:00:00"
    assert end.isoformat() == "2024-04-01T00:00:00"


def test_static_variables_do_not_claim_temporal_reduction():
    source = series(1, 3, 5)
    source["elevation"] = source.red.isel(time=0, drop=True).assign_attrs(
        units="m", cell_methods="area: mean"
    )
    result = source.gs.composite("MS", "mean")
    native = source.resample(time="MS").mean(keep_attrs=True)

    xr.testing.assert_identical(
        result.elevation.drop_vars(["start_datetime", "end_datetime"]), native.elevation
    )
    assert result.elevation.attrs["cell_methods"] == "area: mean"


def test_interpolated_pixels_contribute_without_inventing_observations():
    monthly = _scenes_with_properties().gs.composite("MS", "mean")
    filled = monthly.interpolate_na(dim="time", method="linear")
    result = filled.gs.composite("QS", "mean", properties="stack")
    native = filled.red.resample(time="QS").mean()

    np.testing.assert_allclose(result.red, native)
    assert _cells(result) == [("2024-01-01", "2024-04-01")]
    import json

    # The empty February states no properties, so it adds no stacked entry.
    assert len(json.loads(str(result.stac_properties.values[0]))["constellation"]) == 2


def test_native_sum_distinguishes_all_missing_from_empty_bins():
    source = series(1, 3, 5).astype(float)
    source.red.values[:2] = np.nan
    result = source.gs.composite("MS", "sum")

    np.testing.assert_allclose(result.red[:, 0, 0], [0, np.nan, 5])


def test_composite_band_roundtrips_observation_coordinates(tmp_path):
    source = _scenes_with_properties().gs.composite("MS", "mean")
    band = source.red
    path = tmp_path / "monthly.nc"
    band.to_netcdf(path)
    with xr.open_dataarray(path) as restored:
        xr.testing.assert_allclose(restored, band)
        assert restored.gs.timespan == band.gs.timespan
        assert restored.gs.crs == band.gs.crs


def test_node_composite_uses_inherited_periods_and_scene_properties():
    monthly = _scenes_with_properties().gs.composite("MS", "mean")
    tree = xr.DataTree.from_dict(
        {
            "/": monthly.drop_vars(list(monthly.data_vars)),
            "/optical": monthly.drop_vars(list(monthly.coords)),
        }
    )
    observations = tree["optical"].to_dataset(inherit="all_coords")

    quarterly = observations.gs.composite("QS", "mean", properties="stack")
    expected = monthly.gs.composite("QS", "mean", properties="stack")
    xr.testing.assert_identical(quarterly, expected)


@pytest.mark.parametrize("as_array", [False, True])
def test_composite_interpolates_pixel_gaps_on_named_dimension_lazily(as_array):
    from dask.callbacks import Callback

    data = series(1, 3, 9, dtype="float32").rename(time="acquired")
    data.red.values[0, 0, 0] = np.nan
    data.red.values[1, 0, 0] = np.nan
    data = data.chunk({"acquired": 1})
    if as_array:
        data = data.red
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = data.gs.composite("MS", "mean", dim="acquired", interpolation="linear")
    assert not tasks
    native = data.resample(acquired="MS").mean(keep_attrs=True)
    expected = native.chunk(acquired=-1).interpolate_na(dim="acquired", method="linear")
    actual = result if as_array else result.red
    expected = expected if as_array else expected.red
    np.testing.assert_allclose(actual.compute(), expected.compute(), equal_nan=True)
    assert np.isnan(actual.isel(acquired=0, y=0, x=0).compute())
    assert np.isfinite(actual.isel(acquired=1, y=1, x=1).compute())
    assert result.start_datetime.dims == ("acquired",)
    assert result.start_datetime.values[1] == np.datetime64("2024-02-01", "ns")
    assert actual.attrs["cell_methods"] == "acquired: mean"


def test_composite_interpolation_preserves_observed_properties():
    import json

    data = _scenes_with_properties().gs.mask_and_scale().rename(time="acquired")
    result = data.gs.composite("MS", "mean", dim="acquired", interpolation="nearest")
    assert result.stac_properties.dims == ("acquired",)
    assert json.loads(str(result.stac_properties.values[1])) == {}
    assert result.end_datetime.values[1] == np.datetime64("2024-03-01", "ns")
    assert np.isfinite(result.red.isel(acquired=1)).all()


def test_composite_rejects_properties_on_another_dimension():
    data = series(1, 3, 9, dtype="float32").rename(time="acquired")
    data = data.assign_coords(stac_properties=("y", ["{}", "{}"]))
    with pytest.raises(ValueError, match="acquired"):
        data.gs.composite("MS", "mean", dim="acquired")


@pytest.mark.parametrize("as_array", [False, True])
def test_composite_refuses_interpolation_of_class_codes(as_array):
    data = series(1, 3, 9).gs.write_flags(
        {1: "water", 3: "trees", 9: "cloud"}, target=["red"]
    )
    if as_array:
        data = data.red
    with pytest.raises(ValueError, match="class codes"):
        data.gs.composite("MS", "first", interpolation="linear")


def test_named_dimension_composite_round_trips_coverage(tmp_path):
    data = series(1, 3, 9, dtype="float32").rename(time="acquired")
    result = data.gs.composite("MS", "mean", dim="acquired", interpolation="linear")
    path = tmp_path / "composite.nc"
    result.to_netcdf(path)
    with xr.open_dataset(path) as restored:
        xr.testing.assert_allclose(restored.red, result.red)
        np.testing.assert_array_equal(restored.start_datetime, result.start_datetime)
        assert restored.end_datetime.dims == ("acquired",)


def test_interpolation_does_not_replace_native_sum_zeros_or_extrapolate():
    data = series(1, 3, 9, dtype="float32")
    data.red.values[:2] = np.nan
    summed = data.gs.composite("MS", "sum", interpolation="linear")
    assert (summed.red.isel(time=0) == 0).all()
    assert np.isfinite(summed.red.isel(time=1)).all()

    data.red.values[:] = np.nan
    data.red.values[1] = 3
    # A missing trailing month stays missing even with interpolation selected.
    result = data.gs.composite("MS", "mean", interpolation="linear")
    assert result.red.isel(time=-1).isnull().all()


@pytest.mark.parametrize("as_array", [False, True])
def test_input_cells_are_replaced_not_reduced(as_array):
    data = series(1, 3, 9, dtype="float32")
    data = data.assign_coords(
        start_datetime=("time", OBSERVED - np.timedelta64(1, "D")),
        end_datetime=("time", OBSERVED + np.timedelta64(1, "D")),
    )
    data = data.red if as_array else data

    result = data.gs.composite("MS", "mean")

    assert _cells(result) == [
        ("2024-01-01", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-03-01", "2024-04-01"),
    ]


def test_composite_uses_native_offset_groups_for_pixels_and_cells():
    data = series(1, 3, 9, dtype="float32")
    options = dict(closed="right", label="right", origin="epoch", offset="5D")
    result = data.gs.composite("30D", "mean", **options)
    grouped = data.resample(time="30D", **options)
    np.testing.assert_allclose(result.red, grouped.mean().red, equal_nan=True)
    actual = list(
        zip(result.start_datetime.values, result.end_datetime.values, strict=True)
    )
    assert actual == _pandas_bins(OBSERVED, "30D", **options)
