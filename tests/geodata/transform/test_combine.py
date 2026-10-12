"""Combining several rasters into one."""

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

from geosave.geodata.core.constructors import raster as build_raster
from geosave.geodata.core.stack import stack
from geosave.geodata.transform.combine import concat_time, merge_bands, mosaic
from geosave.geodata.warnings import DroppedAttrsWarning, GeoSaveWarning

from .conftest import build, geobox, grid_of


def test_merge_preserves_band_order_values_and_metadata():
    optical = build(geobox(), labelled=True)
    radar = build(geobox()).rename_vars({"red": "VV"})
    radar["VV"].values[:] = 300
    result = merge_bands([radar, optical])
    assert tuple(result.data_vars) == ("VV", "red", "cls")
    np.testing.assert_array_equal(result.red, optical.red)
    np.testing.assert_array_equal(result.VV, radar.VV)
    assert result.cls.attrs == optical.cls.attrs
    assert result.red.attrs == optical.red.attrs
    assert result.gs.geobox == optical.gs.geobox


@pytest.mark.parametrize(
    "kind", ["grid", "crs", "time", "scalar_time", "timeless", "collision", "dims"]
)
def test_merge_rejects_implicit_alignment_or_broadcasting(kind):
    first = build(geobox())
    second = first.rename_vars({"red": "nir"})
    if kind == "grid":
        second = second.assign_coords(x=second.x + 5)
    elif kind == "crs":
        second = build(geobox(crs="EPSG:32632")).rename_vars({"red": "nir"})
    elif kind == "time":
        second = build(geobox(), start="2025-01-01").rename_vars({"red": "nir"})
    elif kind == "scalar_time":
        first, second = first.isel(time=0), second.isel(time=1)
    elif kind == "timeless":
        second = second.isel(time=0, drop=True)
    elif kind == "collision":
        second = first
    else:
        second = second.transpose("y", "x", "time")
    with pytest.raises(ValueError):
        merge_bands([first, second])


@pytest.mark.parametrize("rasters", [[], [xr.Dataset()]])
def test_merge_requires_nonempty_rasters(rasters):
    with pytest.raises(ValueError, match="non-empty"):
        merge_bands(rasters)


UTM = "EPSG:32633"


def box(
    bbox: tuple[float, float, float, float] = (0, 0, 40, 40),
    *,
    resolution: float = 10,
    crs: str = UTM,
) -> GeoBox:
    """Build one small projected grid."""
    return GeoBox.from_bbox(bbox, crs=crs, resolution=resolution)


def series(
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
    joined = concat_time([series("2024-01-05"), series("2024-02-10")])

    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_nothing_is_reordered_so_the_caller_may_sort_after() -> None:
    march, january = series("2024-03-01"), series("2024-01-05")

    joined = concat_time([march, january])

    assert dates(joined) == ["2024-03-01", "2024-01-05"]
    assert dates(joined.sortby("time")) == ["2024-01-05", "2024-03-01"]


def test_overlapping_and_repeated_labels_lay_down_untouched() -> None:
    joined = concat_time(
        [series("2024-01-01", "2024-02-01"), series("2024-01-15", "2024-01-01")]
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

    joined = concat_time([empty, series("2024-01-05")])

    assert dates(joined) == ["2024-01-05"]


def test_shared_attrs_survive_and_conflicting_ones_are_dropped() -> None:
    first, second = series("2024-01-05"), series("2024-02-10")
    for raster in (first, second):
        raster.attrs.update({"title": "scene", "license": "CC0"})
    second.attrs["title"] = "other"

    with pytest.warns(DroppedAttrsWarning):
        joined = concat_time([first, second])

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
            "'units'",
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
    a = first(series("2024-01-05", nodata=None)[["red"]])
    b = second(series("2024-02-10", nodata=None)[["red"]])

    with pytest.raises(ValueError, match=semantic):
        concat_time([a, b])


def test_incompatible_dataarray_semantics_are_refused() -> None:
    first = series("2024-01-05", nodata=0).red
    second = series("2024-02-10", nodata=255).red

    with pytest.raises(ValueError, match="Nodata"):
        concat_time([first, second])


def test_concat_keeps_the_dtype_and_laziness_of_the_rasters() -> None:
    lazy = concat_time(
        [series("2024-01-05").chunk(), series("2024-02-10").chunk()],
    )

    assert lazy.red.dtype == np.dtype("uint16")
    assert lazy.red.chunks is not None


def test_a_looser_join_pads_what_a_raster_does_not_reach() -> None:
    west = series("2024-01-05", fill=7)
    east = series("2024-02-10", grid=box((20, 0, 60, 40)), fill=9)

    with quiet():
        joined = concat_time([west, east], join="outer")

    assert joined.red.shape == (2, 4, 6)
    assert joined.red.dtype == np.dtype("uint16")
    assert list(joined.red.isel(time=0, y=0).values) == [7, 7, 7, 7, 0, 0]
    assert list(joined.red.isel(time=1, y=0).values) == [0, 0, 9, 9, 9, 9]


def test_an_exact_join_refuses_rasters_of_different_extent() -> None:
    with quiet(), pytest.raises(xr.AlignmentError):
        concat_time(
            [series("2024-01-05"), series("2024-02-10", grid=box((20, 0, 60, 40)))]
        )


def test_padding_refuses_a_variable_carrying_no_fill_value() -> None:
    with pytest.raises(ValueError, match="carries no fill value"):
        concat_time(
            [
                series("2024-01-05", nodata=None),
                series("2024-02-10", grid=box((20, 0, 60, 40)), nodata=None),
            ],
            join="outer",
        )


def test_a_stack_concatenates_group_by_group() -> None:
    first = stack({"s2": series("2024-01-05"), "s1": series("2024-01-05", name="vv")})
    second = stack({"s2": series("2024-02-10"), "s1": series("2024-02-10", name="vv")})

    joined = concat_time([first, second])

    assert joined.gs.groups == ("s2", "s1")
    assert dates(joined.gs.rasters["s2"]) == ["2024-01-05", "2024-02-10"]
    assert dates(joined.gs.rasters["s1"]) == ["2024-01-05", "2024-02-10"]


def test_stacks_naming_different_groups_are_refused() -> None:
    with pytest.raises(ValueError, match="name different rasters"):
        concat_time(
            [stack({"a": series("2024-01-05")}), stack({"b": series("2024-02-10")})]
        )


def test_concatenating_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="pass at least one"):
        concat_time([])


def test_a_dataarray_concatenates_as_a_dataarray() -> None:
    joined = concat_time([series("2024-01-05").red, series("2024-02-10").red])

    assert isinstance(joined, xr.DataArray)
    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_ungeoreferenced_rasters_concatenate_in_pixel_space() -> None:
    def bare(label: str) -> xr.Dataset:
        return xr.Dataset(
            {"red": (("time", "y", "x"), np.zeros((1, 4, 4), "uint16"))},
            coords={"time": np.array([label], "datetime64[ns]")},
        )

    joined = concat_time([bare("2024-01-05"), bare("2024-02-10")])

    assert joined.red.shape == (2, 4, 4)
    assert joined.gs.geobox is None


def test_mixing_placed_and_unplaced_rasters_is_refused() -> None:
    bare = xr.Dataset(
        {"red": (("time", "y", "x"), np.zeros((1, 4, 4), "uint16"))},
        coords={"time": np.array(["2024-02-10"], "datetime64[ns]")},
    )

    with pytest.raises(ValueError, match="EPSG:32633"):
        concat_time([series("2024-01-05"), bare])


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
        concat_time([series("2024-01-05"), series("2024-02-10", grid=other)])


def test_grids_covering_different_ground_are_refused() -> None:
    # Laying these end to end would interleave x labels rather than extend time.
    with pytest.raises(ValueError, match="not equal along these coord"):
        concat_time(
            [series("2024-01-05"), series("2024-02-10", grid=box((40, 0, 80, 40)))]
        )


def test_rasters_placed_by_nothing_still_lay_end_to_end() -> None:
    bare = series("2024-01-05").drop_vars(["y", "x", "spatial_ref"])
    later = series("2024-02-10").drop_vars(["y", "x", "spatial_ref"])

    assert dates(concat_time([bare, later])) == ["2024-01-05", "2024-02-10"]


def test_an_origin_a_hair_off_a_pixel_still_lays_end_to_end() -> None:
    # snap_to and reprojection both land an origin just below zero, which the
    # pixel labels absorb.
    drifted = GeoBox((4, 4), Affine(10.0, 0.0, -4.44e-16, 0.0, -10.0, 40.0), CRS(UTM))

    joined = concat_time([series("2024-01-05"), series("2024-02-10", grid=drifted)])

    assert dates(joined) == ["2024-01-05", "2024-02-10"]


def test_a_grid_lays_end_to_end_with_itself_however_it_is_oriented() -> None:
    south_up = GeoBox((4, 4), Affine(10.0, 0.0, 0.0, 0.0, 10.0, 0.0), CRS(UTM))

    joined = concat_time([series("2024-01-05", grid=south_up)] * 2)

    assert dates(joined) == ["2024-01-05", "2024-01-05"]
    with pytest.raises(ValueError, match="not equal along these coord"):
        concat_time([series("2024-01-05"), series("2024-02-10", grid=south_up)])


def test_root_attrs_every_raster_disagrees_on_are_dropped() -> None:
    first, second = series("2024-01-05"), series("2024-02-10")
    first.attrs, second.attrs = {"provider": "a"}, {"provider": "b"}

    with pytest.warns(DroppedAttrsWarning, match="provider"):
        joined = concat_time([first, second])

    assert "provider" not in joined.attrs


def test_concat_refuses_missing_bands_instead_of_promoting_to_nan() -> None:
    with pytest.raises(ValueError, match="same variables"):
        concat_time([series("2024-01-01"), series("2024-01-02", name="nir")])


def granule(
    grid: GeoBox | None = None,
    *,
    fill: int = 7,
    dtype: str = "uint16",
    nodata: int | None = 0,
    name: str = "red",
) -> xr.Dataset:
    """Build a timeless raster holding `fill` at every pixel."""
    grid = grid or box()
    return build_raster(
        {name: (("y", "x"), np.full(grid.shape, fill, dtype))}, grid, nodata=nodata
    )


def row(raster: xr.Dataset, name: str = "red") -> list[int]:
    """Read a raster's top row as plain integers."""
    return [int(value) for value in raster[name].isel(y=0).values]


def test_rasters_lay_onto_the_ground_they_jointly_cover() -> None:
    with quiet():
        laid = mosaic([granule(), granule(box((20, 0, 60, 40)), fill=9)])

    assert laid.red.shape == (4, 6)
    assert laid.odc.geobox.extent.boundingbox[0] == 0.0
    assert laid.odc.geobox.extent.boundingbox[2] == 60.0


def test_attrs_describe_the_shared_pixels_and_the_output_grid() -> None:
    first = granule()
    second = granule(box((20, 0, 60, 40)))
    for raster in (first, second):
        raster.attrs.update({"license": "CC0", "title": "scene"})
        raster.red.attrs["source_note"] = "shared"
    second.attrs["title"] = "other"
    second.red.attrs["source_note"] = "other"

    with pytest.warns(DroppedAttrsWarning):
        laid = mosaic([first, second])

    assert laid.attrs["license"] == "CC0"
    assert "title" not in laid.attrs
    assert "source_note" not in laid.red.attrs
    assert laid.x.attrs["axis"] == "X"
    assert laid.y.attrs["axis"] == "Y"
    stored = tuple(
        float(value) for value in laid.spatial_ref.attrs["GeoTransform"].split()
    )
    assert Affine.from_gdal(*stored) == grid_of(laid).transform
    assert first.attrs["title"] == "scene"
    assert first.red.attrs["source_note"] == "shared"


def test_the_first_raster_wins_an_overlapping_pixel() -> None:
    with quiet():
        laid = mosaic([granule(fill=7), granule(box((20, 0, 60, 40)), fill=9)])

    assert row(laid) == [7, 7, 7, 7, 9, 9]


def test_the_last_raster_wins_when_it_is_asked_to() -> None:
    with quiet():
        laid = mosaic(
            [granule(fill=7), granule(box((20, 0, 60, 40)), fill=9)], method="last"
        )

    assert row(laid) == [7, 7, 9, 9, 9, 9]


def test_ground_no_raster_covers_holds_the_fill_value() -> None:
    with quiet():
        laid = mosaic([granule(fill=7), granule(box((80, 0, 120, 40)), fill=9)])

    assert laid.red.shape == (4, 12)
    assert row(laid) == [7, 7, 7, 7, 0, 0, 0, 0, 9, 9, 9, 9]


def test_a_nodata_pixel_is_filled_by_a_later_raster() -> None:
    covered = granule(fill=7)
    covered["red"][0, 0] = 0

    with quiet():
        laid = mosaic([covered, granule(fill=9)])

    assert row(laid) == [9, 7, 7, 7]


def test_rasters_with_different_nodata_markers_are_refused() -> None:
    first = granule(nodata=0)
    second = granule(nodata=255)

    with pytest.raises(ValueError, match=r"'red'.*Nodata"):
        mosaic([first, second])


def test_three_rasters_lay_down_in_preference_order() -> None:
    with quiet():
        laid = mosaic(
            [
                granule(fill=7),
                granule(box((20, 0, 60, 40)), fill=8),
                granule(box((40, 0, 80, 40)), fill=9),
            ]
        )

    assert row(laid) == [7, 7, 7, 7, 8, 8, 9, 9]


def test_one_raster_mosaics_to_itself() -> None:
    laid = mosaic([granule(fill=7)])

    assert laid.red.shape == (4, 4)
    assert row(laid) == [7, 7, 7, 7]


def test_mosaic_keeps_the_dtype_and_laziness_of_the_rasters() -> None:
    with quiet():
        laid = mosaic([granule().chunk(), granule(box((20, 0, 60, 40))).chunk()])

    assert laid.red.dtype == np.dtype("uint16")
    assert laid.red.chunks is not None


def test_the_result_carries_the_crs_it_was_laid_on() -> None:
    with quiet():
        laid = mosaic([granule(), granule(box((20, 0, 60, 40)))])

    assert laid.odc.crs is not None
    assert laid.odc.crs.epsg == 32633


def test_mosaicking_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="pass at least one"):
        mosaic([])


def test_a_raster_carrying_no_fill_value_is_refused() -> None:
    with quiet(), pytest.raises(ValueError, match="carries no fill value"):
        mosaic([granule(nodata=None), granule(box((20, 0, 60, 40)), nodata=None)])


def test_rasters_naming_different_variables_are_refused() -> None:
    with pytest.raises(ValueError, match="are not in every raster's variables"):
        mosaic([granule(), granule(name="nir")])


def test_a_later_raster_carrying_an_extra_variable_is_refused() -> None:
    first = granule()
    second = granule(box((20, 0, 60, 40)))
    second["nir"] = second["red"]

    with pytest.raises(ValueError, match=r"\['nir'\] are not in every raster's"):
        mosaic([first, second])


def test_rasters_carrying_no_grid_are_refused() -> None:
    bare = xr.Dataset({"red": (("y", "x"), np.zeros((4, 4), "uint16"))})
    bare["red"].attrs["_FillValue"] = 0

    with pytest.raises(ValueError, match="sits on no locatable grid"):
        mosaic([bare, bare])


@pytest.mark.parametrize(
    ("other", "mismatch"),
    [
        (box(resolution=20), "resolution"),
        (box(crs="EPSG:32748"), "CRS EPSG:32748 against EPSG:32633"),
        (GeoBox((4, 4), Affine(10.0, 0.0, 3.0, 0.0, -10.0, 40.0), CRS(UTM)), "phase"),
    ],
    ids=["resolution", "crs", "pixel phase"],
)
def test_mosaic_refuses_rasters_no_one_grid_holds(other: GeoBox, mismatch: str) -> None:
    with quiet(), pytest.raises(ValueError, match=mismatch):
        mosaic([granule(), granule(other, fill=9)])


def test_mosaic_lays_floats_whose_nodata_is_nan() -> None:
    import numpy as np
    from odc.geo.geobox import GeoBox

    from geosave.geodata.core.constructors import raster as build

    west = GeoBox.from_bbox((0, 0, 80, 80), crs="EPSG:32633", resolution=10)
    east = GeoBox.from_bbox((40, 0, 120, 80), crs="EPSG:32633", resolution=10)
    left = build(
        {"red": (("y", "x"), np.full(west.shape, 5.0, "float32"))},
        west,
        nodata=np.nan,
    )
    right = build(
        {"red": (("y", "x"), np.full(east.shape, 9.0, "float32"))},
        east,
        nodata=np.nan,
    )

    laid = mosaic([left, right])

    assert laid.red.shape == (8, 12)
    assert laid.red.values[0, 0] == 5.0 and laid.red.values[0, -1] == 9.0


def test_mosaic_refuses_different_scalar_observations() -> None:
    first = granule().assign_coords(time=np.datetime64("2024-01-01"))
    second = granule().assign_coords(time=np.datetime64("2024-01-02"))
    with pytest.raises(ValueError, match="coordinates"):
        mosaic([first, second])


def test_merge_bands_ignores_the_grid_mappings_placeholder_value() -> None:
    # Readers disagree on the CF placeholder: rioxarray stores 0, odc-geo the EPSG code.
    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    red = build_raster({"red": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    nir = build_raster({"nir": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    red = red.assign_coords(spatial_ref=red.spatial_ref.copy(data=0))
    nir = nir.assign_coords(spatial_ref=nir.spatial_ref.copy(data=32749))

    merged = merge_bands([red, nir])

    assert list(merged.data_vars) == ["red", "nir"]
    assert merged.gs.geobox == box


def test_merge_bands_keeps_one_spelling_of_the_grid_coordinates() -> None:
    # A band sliced from a file has bare x/y; one odc-geo warped has CF attrs.
    box = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    red = build_raster({"red": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    nir = build_raster({"nir": (("y", "x"), np.ones((2, 2), "uint16"))}, box)
    red = red.copy()
    red.x.attrs, red.y.attrs = {}, {}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DroppedAttrsWarning)
        merged = merge_bands([red, nir])

    assert merged.x.attrs["axis"] == "X"
    assert merged.y.attrs["standard_name"] == "projection_y_coordinate"
