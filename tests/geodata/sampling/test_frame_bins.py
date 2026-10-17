"""Frame requests survive persistence and execute lazily on saved rasters."""

import warnings

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

import geosave as gs
from geosave.geodata.warnings import DroppedRowsWarning


def catalog(tmp_path, scenes, *, nodata=np.nan):
    grid = GeoBox.from_bbox((0, 0, 20, 20), "EPSG:3857", resolution=10)
    entries = []
    for index, (name, date, value, bounds) in enumerate(scenes):
        data = gs.raster(
            {"value": (("y", "x"), da.full((2, 2), float(value), chunks=(2, 2)))},
            grid,
            nodata=nodata,
        )
        if date is not None:
            data = data.assign_coords(time=np.datetime64(date, "us"))
        if bounds:
            data = data.assign_coords(
                start_datetime=np.datetime64(bounds[0], "us"),
                end_datetime=np.datetime64(bounds[1], "us"),
            )
        path = tmp_path / f"{index}.tif"
        data.gs.to_cog(path)
        entries.extend(
            gs.stac.create_items({"value": path}, collection=name, id_format=str(index))
        )
    return gs.stac.create_catalog(entries)


def test_weekly_composites_and_reference_survive_parquet(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, ("2024-01-01", "2024-01-07T23:59:59.999999")),
            ("image", "2024-01-08", 4, ("2024-01-08", "2024-01-14T23:59:59.999999")),
            ("label", "2024-01-15", 9, None),
        ],
    )
    frames = rows.gs.sample().gs.frame(
        windows={"image": {"slots": ["-2W", "-1W"]}}, reference="label"
    )
    assert len(frames) == 1
    restored = gs.open_vector(frames.gs.to_geoparquet(tmp_path / "frames.parquet"))
    tree = restored.iloc[0].gs.to_xarray()
    image = tree["image"].dataset
    assert image.sizes["time"] == 2
    assert image.value.chunks is not None
    np.testing.assert_allclose(image.value[:, 0, 0].values, [2, 4])
    np.testing.assert_allclose(tree["label"].dataset.value.values, 9)
    np.testing.assert_array_equal(
        image.start_datetime.values,
        np.array(["2024-01-01", "2024-01-08"], dtype="datetime64[us]"),
    )


def test_reduce_first_selects_whole_acquisition_and_interpolates_missing_bin(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, None),
            ("image", "2024-01-02", 99, None),
            ("image", "2024-01-15", 6, None),
            ("label", "2024-01-22", 9, None),
        ],
    )
    frames = rows.gs.sample().gs.frame(
        windows={
            "image": {
                "slots": ["-3W", "-2W", "-1W"],
                "reduce": "first",
                "fill": "interpolate",
            }
        },
        reference="label",
    )
    image = frames.iloc[0].gs.to_xarray()["image"].dataset
    assert image.value.chunks is not None
    np.testing.assert_allclose(image.value[:, 0, 0].values, [2, 4, 6])


def test_crossing_composite_is_not_a_weekly_observation(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, ("2024-01-01", "2024-01-31T23:59:59.999999")),
            ("label", "2024-01-15", 9, None),
        ],
    )
    with pytest.warns(DroppedRowsWarning):
        frames = rows.gs.sample().gs.frame(
            windows={"image": {"slots": ["-1W"]}}, reference="label"
        )
    assert frames.empty


def test_default_reference_is_newest_first_and_stride_counts_candidates(tmp_path):
    rows = catalog(tmp_path, [("image", f"2024-01-0{i}", i, None) for i in range(1, 6)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DroppedRowsWarning)
        frames = rows.gs.sample().gs.frame(windows={"image": {"slots": ["-2D", "-1D"]}})
    assert frames.reference_time.tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-03"),
    ]
    np.testing.assert_allclose(
        frames.iloc[0].gs.to_xarray()["image"].dataset.value[:, 0, 0], [3, 4]
    )


def test_multiple_observations_require_reducer(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 1, None),
            ("image", "2024-01-02", 2, None),
            ("label", "2024-01-08", 9, None),
        ],
    )
    with pytest.raises(ValueError, match="reduce"):
        rows.gs.sample().gs.frame(
            windows={"image": {"slots": ["-1W"]}}, reference="label"
        )


def test_interpolation_does_not_blend_nodata_sentinels(tmp_path):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", -9999, None),
            ("image", "2024-01-15", 5, None),
            ("label", "2024-01-22", 9, None),
        ],
        nodata=-9999,
    )
    samples = rows.gs.sample()
    frames = samples.gs.frame(
        windows={"image": {"slots": ["-3W", "-2W", "-1W"], "fill": "interpolate"}},
        reference="label",
    )
    image = frames.iloc[0].gs.to_xarray()["image"].dataset
    assert np.isnan(image.value.isel(time=1).values).all()


def test_timeless_auxiliary_survives_framing(tmp_path):
    rows = catalog(
        tmp_path, [("image", "2024-01-01", 1, None), ("dem", None, 10, None)]
    )
    frame = rows.gs.sample().gs.frame(windows={"image": {"slots": ["0D"]}}).iloc[0]
    dem = frame.gs.to_xarray()["dem"].dataset
    assert "time" not in dem.dims
    np.testing.assert_allclose(dem.value.values, 10)


@pytest.mark.parametrize("method,expected", [("first", 2), ("last", 6), ("mean", 4)])
def test_same_timestamp_acquisitions_reduce_as_separate_observations(
    tmp_path, method, expected
):
    rows = catalog(
        tmp_path,
        [
            ("image", "2024-01-01", 2, None),
            ("image", "2024-01-01", 6, None),
            ("label", "2024-01-08", 9, None),
        ],
    )
    frame = (
        rows.gs.sample()
        .gs.frame(
            windows={"image": {"slots": ["-1W"], "reduce": method}}, reference="label"
        )
        .iloc[0]
    )
    np.testing.assert_allclose(
        frame.gs.to_xarray()["image"].dataset.value.values, expected
    )
