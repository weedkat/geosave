import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave.geodata import CONTEXT_READERS
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

    assert CONTEXT_READERS["year"](tree).tolist() == [2024, 2024]
    assert CONTEXT_READERS["day_of_year"](tree).tolist() == [366, 1]
    assert CONTEXT_READERS["week"](tree).tolist() == [1, 1]
    assert CONTEXT_READERS["hour"](tree).tolist() == [14, 3]


def test_times_are_nanoseconds_whatever_the_stored_unit():
    tree = build_scene()

    time = CONTEXT_READERS["time"](tree)

    assert time.dtype == np.int64
    restored = pd.to_datetime(time)
    assert restored.equals(pd.DatetimeIndex(["2024-12-31T14:00", "2024-01-01T03:00"]))


def test_the_centroid_is_the_grids_centre_in_degrees():
    centroid = CONTEXT_READERS["centroid"](build_scene())

    assert centroid.dtype == np.float32
    np.testing.assert_allclose(centroid, [51.0, 11.0])


def test_one_reader_serves_a_tree_a_dataset_and_a_dataarray():
    scene = build_raster(times=2)
    rasters = [xr.DataTree.from_dict({"image": scene}), scene, scene.red]

    for raster in rasters:
        assert CONTEXT_READERS["year"](raster).tolist() == [2025, 2025]
        assert CONTEXT_READERS["crs"](raster).item() == 32749
        transform = CONTEXT_READERS["transform"](raster)
        assert transform.dtype == np.float64
        assert transform.tolist() == list(scene.gs.geobox.transform[:6])


def test_one_observation_is_dated_by_its_scalar_time():
    observation = build_raster(times=1).isel(time=0)
    tree = xr.DataTree.from_dict({"image": observation, "label": build_raster()})

    for raster in [observation, observation.red, tree]:
        assert CONTEXT_READERS["day_of_year"](raster).tolist() == [152]


def test_time_context_needs_one_timeline():
    scene = build_raster(times=2)
    reversed_scene = scene.isel(time=[1, 0])

    with pytest.raises(ValueError, match="one timeline"):
        CONTEXT_READERS["time"](
            xr.DataTree.from_dict({"a": scene, "b": reversed_scene})
        )
    with pytest.raises(ValueError, match="one timeline"):
        CONTEXT_READERS["time"](
            xr.DataTree.from_dict({"a": scene.isel(time=0), "b": scene.isel(time=1)})
        )
    with pytest.raises(ValueError, match="datetime `time`"):
        CONTEXT_READERS["year"](xr.DataTree.from_dict({"image": build_raster()}))


def test_grid_context_needs_an_epsg_code():
    custom = "+proj=utm +zone=49 +south +ellps=GRS80 +towgs84=1,2,3 +units=m"

    with pytest.raises(ValueError, match="EPSG code"):
        CONTEXT_READERS["crs"](build_raster(crs=custom))


def test_to_tensor_reads_named_and_project_readers_under_the_callers_keys():
    def read_band_count(raster: xr.Dataset) -> np.ndarray:
        return np.array(len(raster.data_vars))

    _, context = build_raster(times=2).gs.to_tensor(
        context={"doy": "day_of_year", "bands": read_band_count}
    )

    assert list(context) == ["doy", "bands"]
    assert isinstance(context["doy"], torch.Tensor)
    assert context["doy"].tolist() == [152, 153]
    assert context["bands"].item() == 2
    with pytest.raises(KeyError, match="'elevation' is not one of"):
        build_raster().gs.to_tensor(context={"elevation": "elevation"})
