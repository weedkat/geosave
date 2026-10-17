from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo import CRS
from odc.geo.geobox import GeoBox

import geosave.geodata.attrs as attrs
from geosave.geodata.core.raster.create import raster as build_raster
from geosave.geodata.core.raster.create import tree
from geosave.geodata.transform.temporal import concat, interpolate, reduce, resample
from geosave.geodata.warnings import DroppedAttrsWarning, GeoSaveWarning
from tests.geodata.conftest import geobox

UTM = "EPSG:32633"

# Two observations in January, one in March, so February covers none.
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


def test_the_bucketing_survives_the_reduction() -> None:
    # xarray carries coordinate attrs through a groupby reduce, which is the only
    # reason a reduced axis still says what its labels stand for.
    monthly = reduce(resample(series(100, 900, 500), "ME"), "median")

    bucketing = attrs.TimeSpec.from_attrs(monthly.time.attrs)

    assert bucketing is not None
    assert bucketing.time_freq == "ME"
    assert (bucketing.time_closed, bucketing.time_label) == ("right", "right")


def test_the_result_is_the_kind_the_resampler_holds() -> None:
    scene = series(100, 900, 500)

    assert isinstance(reduce(resample(scene, "MS"), "median"), xr.Dataset)
    assert isinstance(reduce(resample(scene.red, "MS"), "median"), xr.DataArray)


def test_flag_values_refuse_a_reducer_that_ranks_them() -> None:
    # min returns a real class code, which is why it reads as safe and is not:
    # it asserts an order over codes that carry none.
    coded = series(4, 5, 4, dtype="uint8").gs.write_legend(
        {4: "veg", 5: "water"}, target="red"
    )

    with pytest.raises(ValueError, match="class codes"):
        reduce(resample(coded, "MS"), "min")


def test_a_flag_variable_alone_refuses_the_same_reducer() -> None:
    coded = series(4, 5, 4, dtype="uint8").gs.write_legend(
        {4: "veg", 5: "water"}, target="red"
    )

    with pytest.raises(ValueError, match="class codes"):
        reduce(resample(coded.red, "MS"), "min")


def test_flag_values_take_a_point_sample() -> None:
    coded = series(4, 5, 4, dtype="uint8").gs.write_legend(
        {4: "veg", 5: "water"}, target="red"
    )

    sampled = reduce(resample(coded, "MS"), "first")

    assert sampled.red.values[0, 0, 0] == 4
    assert sampled.red.attrs["cell_methods"] == "time: point"


def test_the_nodata_value_is_left_out_of_the_reduction() -> None:
    # January reads its nodata value and 900, so the month observed only 900.
    scene = series(0, 900, 500, nodata=0)

    assert reduce(resample(scene, "MS"), "mean").red.values[0, 0, 0] == 900


def test_keeping_the_nodata_value_blends_it_in() -> None:
    scene = series(0, 900, 500, nodata=0)

    kept = reduce(resample(scene, "MS", skip_nodata=False), "mean")

    assert kept.red.values[0, 0, 0] == 450


def monthly(*values: int) -> xr.Dataset:
    """Collapse `series` into monthly buckets, February observing none."""
    return reduce(resample(series(*values), "MS"), "median")


def test_a_bucket_observing_nothing_holds_its_label_and_absence() -> None:
    buckets = monthly(100, 100, 900)

    # resample cuts every bucket its cadence covers, so nothing is missing from
    # the axis; February is present and absent throughout.
    assert [str(label)[:7] for label in buckets.time.values] == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]
    assert bool(buckets.red.isel(time=1).isnull().all())


def test_nearest_repeats_the_nearest_observed_bucket() -> None:
    filled = interpolate(monthly(100, 100, 900), "nearest")

    # February sits 31 days after January and 29 before March, so March is nearer.
    assert filled.red.values[:, 0, 0].tolist() == [100.0, 900.0, 900.0]


def test_interpolate_stays_lazy_on_a_time_axis_split_in_chunks() -> None:
    import dask.array as da

    buckets = monthly(100, 100, 900).chunk(time=1)

    filled = interpolate(buckets, "nearest")

    assert isinstance(filled.red.data, da.Array)
    assert filled.red.values[:, 0, 0].tolist() == [100.0, 900.0, 900.0]


def test_linear_reads_between_the_observed_buckets() -> None:
    filled = interpolate(monthly(100, 100, 900), "linear")

    # Weighted by real elapsed days, not by bucket count.
    assert filled.red.values[1, 0, 0] == pytest.approx(513.333, abs=0.01)


def quarterly() -> xr.Dataset:
    """Collapse four months of readings, each pixel reading 100, 200, 300, 400."""
    box = geobox(shape=(2, 2))
    stamps = np.array(
        ["2024-01-05", "2024-02-05", "2024-03-05", "2024-04-05"], "datetime64[ns]"
    )
    pixels = np.stack(
        [np.full(box.shape, value, "uint16") for value in (100, 200, 300, 400)]
    )
    return reduce(
        resample(
            build_raster(
                {"red": (("time", *("y", "x")), pixels)},
                box,
                time=stamps,
            ),
            "MS",
        ),
        "median",
    )


def test_a_partly_observed_bucket_fills_only_the_cells_it_missed() -> None:
    clouded = quarterly()
    clouded.red.values[1, 0, 0] = np.nan

    filled = interpolate(clouded, "nearest")

    # February sits 31 days after January and 29 before March, so March is nearer.
    assert filled.red.values[1, 0, 0] == 300.0
    assert filled.red.values[1, 1, 1] == 200.0


def test_a_cell_observed_once_stays_absent() -> None:
    lonely = quarterly()
    lonely.red.values[:3, 0, 0] = np.nan

    filled = interpolate(lonely, "nearest")

    # Interpolating reads between two observations, and this cell has one.
    assert bool(np.isnan(filled.red.values[:3, 0, 0]).all())
    assert filled.red.values[0, 1, 1] == 100.0


def test_interpolating_one_variable_returns_a_data_array() -> None:
    buckets = reduce(resample(series(100, 100, 900).red, "MS"), "median")

    filled = interpolate(buckets, "nearest")

    assert isinstance(filled, xr.DataArray)
    assert filled.values[:, 0, 0].tolist() == [100.0, 900.0, 900.0]


def test_linear_refuses_to_blend_class_codes() -> None:
    coded = monthly(4, 4, 5).gs.write_legend({4: "veg", 5: "water"}, target="red")

    with pytest.raises(ValueError, match="class codes"):
        interpolate(coded, "linear")

    # Repeating a code names a class, so the nearest reading is allowed.
    assert interpolate(coded, "nearest").red.values[1, 0, 0] == 5


def test_an_axis_of_instants_refuses_to_be_filled() -> None:
    with pytest.raises(ValueError, match="names instants rather than buckets"):
        interpolate(series(100, 100, 900), "nearest")


def test_a_raster_without_a_time_axis_refuses_to_be_filled() -> None:
    timeless = series(100, 100, 900).isel(time=0, drop=True)

    with pytest.raises(ValueError, match="carries no 'time' coordinate"):
        interpolate(timeless, "nearest")


def test_the_bucketing_survives_the_filling() -> None:
    filled = interpolate(monthly(100, 100, 900), "nearest")

    bucketing = attrs.TimeSpec.from_attrs(filled.time.attrs)

    assert bucketing is not None
    assert bucketing.time_freq == "MS"


def box(
    bbox: tuple[float, float, float, float] = (0, 0, 40, 40),
    *,
    resolution: float = 10,
    crs: str = UTM,
) -> GeoBox:
    """Build one small projected grid."""
    return GeoBox.from_bbox(bbox, crs=crs, resolution=resolution)


def observation(
    *observed: str,
    grid: GeoBox | None = None,
    fill: int = 7,
    dtype: str = "uint16",
    nodata: int | None = 0,
    name: str = "red",
) -> xr.Dataset:
    """Build a raster observing `fill` at each label in `observed`."""
    grid = grid or box()
    labels = np.array(observed, "datetime64[ns]")
    pixels = np.stack([np.full(grid.shape, fill, dtype) for _ in labels])
    return build_raster(
        {name: (("time", *("y", "x")), pixels)},
        grid,
        time=labels,
        nodata=nodata,
    )


def dates(raster: xr.Dataset | xr.DataArray) -> list[str]:
    """Read a raster's time labels as `YYYY-MM-DD` strings."""
    return [str(label)[:10] for label in raster.coords["time"].values]


@contextmanager
def quiet() -> Iterator[None]:
    """Ignore warnings a join raises beside whatever is under test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoSaveWarning)
        yield


def test_rasters_lay_down_in_the_order_they_were_given() -> None:
    joined = concat([observation("2024-01-05"), observation("2024-02-10")])

    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_nothing_is_reordered_so_the_caller_may_sort_after() -> None:
    march, january = observation("2024-03-01"), observation("2024-01-05")

    joined = concat([march, january])

    assert dates(joined) == ["2024-03-01", "2024-01-05"]
    assert dates(joined.sortby("time")) == ["2024-01-05", "2024-03-01"]


def test_overlapping_and_repeated_labels_lay_down_untouched() -> None:
    joined = concat(
        [
            observation("2024-01-01", "2024-02-01"),
            observation("2024-01-15", "2024-01-01"),
        ]
    )

    assert dates(joined) == ["2024-01-01", "2024-02-01", "2024-01-15", "2024-01-01"]


def test_an_unlabelled_or_empty_axis_lays_down_all_the_same() -> None:
    grid = box()
    empty = build_raster(
        {"red": (("time", *("y", "x")), np.zeros((0, *grid.shape), "uint16"))},
        grid,
        time=np.array([], "datetime64[ns]"),
        nodata=0,
    )

    joined = concat([empty, observation("2024-01-05")])

    assert dates(joined) == ["2024-01-05"]


def test_shared_attrs_survive_and_conflicting_ones_are_dropped() -> None:
    first, second = observation("2024-01-05"), observation("2024-02-10")
    for raster in (first, second):
        raster.attrs.update({"title": "scene", "license": "CC0"})
    second.attrs["title"] = "other"

    with pytest.warns(DroppedAttrsWarning):
        joined = concat([first, second])

    assert joined.attrs["license"] == "CC0"
    assert "title" not in joined.attrs


@pytest.mark.parametrize(
    ("first", "second", "semantic"),
    [
        (lambda r: r.gs.write_nodata(0), lambda r: r.gs.write_nodata(255), "Nodata"),
        (
            lambda r: r.gs.write_packing(0.1),
            lambda r: r.gs.write_packing(0.01),
            "Packing",
        ),
        (
            lambda r: r.assign(red=r.red.assign_attrs(units="m")),
            lambda r: r.assign(red=r.red.assign_attrs(units="ft")),
            "Quantity",
        ),
        (
            lambda r: r.gs.write_legend({7: "forest"}),
            lambda r: r.gs.write_legend({7: "water"}),
            "Legend",
        ),
    ],
    ids=["nodata", "packing", "units", "legend"],
)
def test_incompatible_pixel_semantics_are_refused(first, second, semantic: str) -> None:
    a = first(observation("2024-01-05", nodata=None)[["red"]])
    b = second(observation("2024-02-10", nodata=None)[["red"]])

    with pytest.raises(ValueError, match=semantic):
        concat([a, b])


def test_incompatible_dataarray_semantics_are_refused() -> None:
    first = observation("2024-01-05", nodata=0).red
    second = observation("2024-02-10", nodata=255).red

    with pytest.raises(ValueError, match="Nodata"):
        concat([first, second])


def test_concat_keeps_the_dtype_and_laziness_of_the_rasters() -> None:
    lazy = concat(
        [observation("2024-01-05").chunk(), observation("2024-02-10").chunk()],
    )

    assert lazy.red.dtype == np.dtype("uint16")
    assert lazy.red.chunks is not None


def test_a_looser_join_pads_what_a_raster_does_not_reach() -> None:
    west = observation("2024-01-05", fill=7)
    east = observation("2024-02-10", grid=box((20, 0, 60, 40)), fill=9)

    with quiet():
        joined = concat([west, east], join="outer")

    assert joined.red.shape == (2, 4, 6)
    assert joined.red.dtype == np.dtype("uint16")
    assert list(joined.red.isel(time=0, y=0).values) == [7, 7, 7, 7, 0, 0]
    assert list(joined.red.isel(time=1, y=0).values) == [0, 0, 9, 9, 9, 9]


def test_an_exact_join_refuses_rasters_of_different_extent() -> None:
    with quiet(), pytest.raises(xr.AlignmentError):
        concat(
            [
                observation("2024-01-05"),
                observation("2024-02-10", grid=box((20, 0, 60, 40))),
            ]
        )


def test_padding_refuses_a_variable_carrying_no_fill_value() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        concat(
            [
                observation("2024-01-05", nodata=None),
                observation("2024-02-10", grid=box((20, 0, 60, 40)), nodata=None),
            ],
            join="outer",
        )


def test_a_tree_concatenates_group_by_group() -> None:
    first = tree(
        {"s2": observation("2024-01-05"), "s1": observation("2024-01-05", name="vv")}
    )
    second = tree(
        {"s2": observation("2024-02-10"), "s1": observation("2024-02-10", name="vv")}
    )

    joined = concat([first, second])

    assert joined.gs.groups == ("s2", "s1")
    assert dates(joined.gs.rasters["s2"]) == ["2024-01-05", "2024-02-10"]
    assert dates(joined.gs.rasters["s1"]) == ["2024-01-05", "2024-02-10"]


def test_stacks_naming_different_groups_are_refused() -> None:
    with pytest.raises(ValueError, match="name different rasters"):
        concat(
            [
                tree({"a": observation("2024-01-05")}),
                tree({"b": observation("2024-02-10")}),
            ]
        )


def test_concatenating_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="pass at least one"):
        concat([])


def test_a_dataarray_concatenates_as_a_dataarray() -> None:
    joined = concat([observation("2024-01-05").red, observation("2024-02-10").red])

    assert isinstance(joined, xr.DataArray)
    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_ungeoreferenced_rasters_concatenate_in_pixel_space() -> None:
    def bare(label: str) -> xr.Dataset:
        return xr.Dataset(
            {"red": (("time", "y", "x"), np.zeros((1, 4, 4), "uint16"))},
            coords={"time": np.array([label], "datetime64[ns]")},
        )

    joined = concat([bare("2024-01-05"), bare("2024-02-10")])

    assert joined.red.shape == (2, 4, 4)
    assert joined.gs.geobox is None


def test_mixing_placed_and_unplaced_rasters_is_refused() -> None:
    bare = xr.Dataset(
        {"red": (("time", "y", "x"), np.zeros((1, 4, 4), "uint16"))},
        coords={"time": np.array(["2024-02-10"], "datetime64[ns]")},
    )

    with pytest.raises(ValueError, match="EPSG:32633"):
        concat([observation("2024-01-05"), bare])


@pytest.mark.parametrize(
    ("other", "mismatch"),
    [
        (box(resolution=20), "not equal along these coord"),
        (box(crs="EPSG:32748"), "EPSG:32748"),
        (
            GeoBox((4, 4), Affine(10.0, 0.0, 3.0, 0.0, -10.0, 40.0), CRS(UTM)),
            "not equal along these coord",
        ),
    ],
    ids=["resolution", "crs", "pixel phase"],
)
def test_concat_refuses_rasters_no_one_grid_holds(other: GeoBox, mismatch: str) -> None:
    with pytest.raises(ValueError, match=mismatch):
        concat([observation("2024-01-05"), observation("2024-02-10", grid=other)])


def test_grids_covering_different_ground_are_refused() -> None:
    # Laying these end to end would interleave x labels rather than extend time.
    with pytest.raises(ValueError, match="not equal along these coord"):
        concat(
            [
                observation("2024-01-05"),
                observation("2024-02-10", grid=box((40, 0, 80, 40))),
            ]
        )


def test_rasters_placed_by_nothing_still_lay_end_to_end() -> None:
    bare = observation("2024-01-05").drop_vars(["y", "x", "spatial_ref"])
    later = observation("2024-02-10").drop_vars(["y", "x", "spatial_ref"])

    assert dates(concat([bare, later])) == ["2024-01-05", "2024-02-10"]


def test_an_origin_a_hair_off_a_pixel_still_lays_end_to_end() -> None:
    # snap_to and reprojection both land an origin just below zero, which the
    # pixel labels absorb.
    drifted = GeoBox((4, 4), Affine(10.0, 0.0, -4.44e-16, 0.0, -10.0, 40.0), CRS(UTM))

    joined = concat(
        [observation("2024-01-05"), observation("2024-02-10", grid=drifted)]
    )

    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_a_grid_lays_end_to_end_with_itself_however_it_is_oriented() -> None:
    south_up = GeoBox((4, 4), Affine(10.0, 0.0, 0.0, 0.0, 10.0, 0.0), CRS(UTM))

    joined = concat([observation("2024-01-05", grid=south_up)] * 2)

    assert dates(joined) == ["2024-01-05", "2024-01-05"]
    with pytest.raises(ValueError, match="not equal along these coord"):
        concat([observation("2024-01-05"), observation("2024-02-10", grid=south_up)])


def test_root_attrs_every_raster_disagrees_on_are_dropped() -> None:
    first, second = observation("2024-01-05"), observation("2024-02-10")
    first.attrs, second.attrs = {"provider": "a"}, {"provider": "b"}

    with pytest.warns(DroppedAttrsWarning, match="provider"):
        joined = concat([first, second])

    assert "provider" not in joined.attrs


def test_concat_refuses_missing_bands_instead_of_promoting_to_nan() -> None:
    with pytest.raises(ValueError, match="same variables"):
        concat([observation("2024-01-01"), observation("2024-01-02", name="nir")])


@pytest.mark.parametrize("as_array", [False, True])
def test_select_keeps_overlapping_composites_lazy_and_in_source_order(as_array):
    import dask.array as da
    from geosave.geodata.transform import temporal

    data = series(1, 2, 3).chunk({"time": 1})
    data = data.assign_coords(
        start_datetime=(
            "time",
            np.array(
                ["2024-01-01", "2024-01-10", "2024-03-01"], dtype="datetime64[ns]"
            ),
        ),
        end_datetime=(
            "time",
            np.array(
                ["2024-01-31", "2024-01-20", "2024-03-31"], dtype="datetime64[ns]"
            ),
        ),
    )
    if as_array:
        data = data.red
    selected = temporal.select(data, "2024-01-25")
    assert selected.sizes["time"] == 1
    np.testing.assert_array_equal(selected.time, data.time[:1])
    values = selected if as_array else selected.red
    assert isinstance(values.data, da.Array)
    np.testing.assert_array_equal(values.compute(), 1)


def test_select_handles_scalar_composite_bounds_after_time_expansion():
    from geosave.geodata.transform import temporal

    data = (
        series(1, 2, 3)
        .isel(time=0)
        .assign_coords(
            start_datetime=np.datetime64("2024-01-01"),
            end_datetime=np.datetime64("2024-01-31"),
        )
        .expand_dims("time")
    )
    assert temporal.select(data, "2024-01-25").sizes["time"] == 1
    assert temporal.select(data, "2024-02").sizes["time"] == 0


def test_select_tree_keeps_static_groups_and_filters_each_time_axis():
    from geosave.geodata.transform import temporal

    data = tree(
        {"dated": series(1, 2, 3), "static": series(4, 5, 6).isel(time=0, drop=True)}
    )
    selected = temporal.select(data, "2024-01")
    assert selected["dated"].dataset.sizes["time"] == 2
    assert selected["static"].dataset.identical(data["static"].dataset)
    assert temporal.select(data, None) is data


@pytest.mark.parametrize("as_array", [False, True])
def test_concat_requires_explicit_dtype_conversion(as_array):
    first = series(1, 2, 3)
    second = first.astype("float32")
    if as_array:
        first, second = first.red, second.red

    with pytest.raises(ValueError, match="dtype"):
        concat([first, second])


def test_concat_requires_a_time_coordinate():
    timeless = series(1, 2, 3).isel(time=0, drop=True)

    with pytest.raises(ValueError, match="time coordinate"):
        concat([timeless, timeless])


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_categorical_validation_precedes_reduction(kind, monkeypatch):
    coded = series(4, 5, 4, dtype="uint8").gs.write_legend(
        {4: "veg", 5: "water"}, target="red"
    )
    data = coded.red if kind == "array" else coded
    grouped = resample(data, "MS")

    def refuse_reduction(*args, **kwargs):
        pytest.fail("categorical pixels were reduced before validation")

    monkeypatch.setattr(type(grouped), "mean", refuse_reduction)
    with pytest.raises(ValueError, match="class codes"):
        reduce(grouped, "mean")


@pytest.mark.parametrize("kind", ["array", "dataset"])
def test_reduction_preserves_metadata_under_xarray_drop_attrs_option(kind):
    from dask.callbacks import Callback

    source = series(100, 900, 500).chunk({"time": 1})
    source.red.attrs["units"] = "m"
    data = source.red if kind == "array" else source
    grouped = resample(data, "MS")

    with (
        xr.set_options(keep_attrs=False),
        Callback(pretask=lambda *_: pytest.fail("reduction computed lazy pixels")),
    ):
        result = reduce(grouped, "mean")

    variable = result if kind == "array" else result.red
    assert variable.attrs["units"] == "m"
    assert variable.attrs["cell_methods"] == "time: mean"
    assert attrs.TimeSpec.from_attrs(result.time.attrs).time_freq == "MS"
    assert variable.chunks is not None
    np.testing.assert_allclose(variable.compute()[:, 0, 0], [500, np.nan, 500])
