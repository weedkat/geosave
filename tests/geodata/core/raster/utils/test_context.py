import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata import TENSOR_CONTEXT
from tests.conftest import build_dataset
from tests.geodata.conftest import build_raster

torch = pytest.importorskip("torch")


def build_scene() -> xr.DataTree:
    grid = GeoBox.from_bbox((10, 50, 12, 52), "EPSG:4326", shape=(8, 8))
    scene = build_dataset(
        {"red": (("time", "y", "x"), np.zeros((2, 8, 8), dtype="float32"))},
        grid,
        time=np.array(["2024-12-31T14:00", "2024-01-01T03:00"], dtype="datetime64[m]"),
    )
    return xr.DataTree.from_dict({"image": scene})


def test_dates_are_read_per_frame_in_axis_order():
    tree = build_scene()

    assert TENSOR_CONTEXT["year"](tree).tolist() == [2024, 2024]
    assert TENSOR_CONTEXT["day_of_year"](tree).tolist() == [366, 1]
    assert TENSOR_CONTEXT["week"](tree).tolist() == [1, 1]
    assert TENSOR_CONTEXT["hour"](tree).tolist() == [14, 3]


def test_times_are_nanoseconds_whatever_the_stored_unit():
    tree = build_scene()

    time = TENSOR_CONTEXT["time"](tree)

    assert time.dtype == torch.int64
    restored = pd.to_datetime(time.numpy())
    assert restored.equals(pd.DatetimeIndex(["2024-12-31T14:00", "2024-01-01T03:00"]))


def test_the_centroid_is_the_grids_centre_in_degrees():
    centroid = TENSOR_CONTEXT["centroid"](build_scene())

    assert centroid.dtype == torch.float32
    torch.testing.assert_close(centroid, torch.tensor([51.0, 11.0]))


def test_time_context_needs_one_timeline():
    scene = build_raster(times=2)
    reversed_scene = scene.isel(time=[1, 0])

    with pytest.raises(ValueError, match="one timeline"):
        TENSOR_CONTEXT["time"](xr.DataTree.from_dict({"a": scene, "b": reversed_scene}))
    with pytest.raises(ValueError, match="datetime `time`"):
        TENSOR_CONTEXT["year"](xr.DataTree.from_dict({"image": build_raster()}))
