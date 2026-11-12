"""Small real raster operations with synthetic acquisition and a toy model."""

import dask.array as da
from odc.geo.geobox import GeoBox
import torch
import xarray as xr

import geosave as gs
from geosave.geodata.io.formats.geotiff import write_cog
from tests.conftest import place


def scene(value=2.0):
    grid = GeoBox.from_bbox(
        (500000, 9000000, 500040, 9000040), "EPSG:32748", resolution=10
    )
    pixels = da.full((4, 4), value, chunks=(2, 2), dtype="float32")
    return xr.Dataset({"red": (("y", "x"), pixels)}).pipe(place, grid)


def accept(raster):
    return raster


def open_raster(path):
    # This experiment owns and closes the file; a streaming recipe must instead
    # keep the resource open across its complete consumer lifetime.
    with gs.open_raster(path) as raster:
        return raster.load()


def acquire_area(geometry, time):
    region = gs.GeoRegion.from_geometry(geometry, crs="EPSG:4326", timespan=time)
    raster = scene()
    return raster.assign_attrs(request_bounds=tuple(region.extent.boundingbox))


def acquire_center(longitude, latitude, width_m, height_m, time):
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Width and height must be positive metres")
    region = gs.GeoRegion.from_coordinates(latitude, longitude, timespan=time)
    return scene().assign_attrs(request_center=region.lonlat)


def validate(raster):
    if not isinstance(raster, xr.Dataset) or list(raster.data_vars) != ["red"]:
        raise ValueError("Expected one red variable")
    raster.gs.validate()
    if raster.red.dims != ("y", "x"):
        raise ValueError("Expected a selected two-dimensional observation")
    return raster


def prepare(raster):
    return raster * 0.5


def predict(raster, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pixels = raster.red.to_numpy()
    model = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[[[1.0]]], [[[-1.0]]]]))
        image = torch.from_numpy(pixels.copy())[None, None]
        scores = model(image)[0].numpy()
    # This tiny model runs one image; batch_size exercises argument routing only.
    return xr.DataArray(scores, dims=("class", "y", "x")).pipe(place, raster.gs.geobox)


def classify(scores):
    return scores.argmax("class").astype("uint8").rename("label").to_dataset()


def save(raster, path):
    return str(write_cog(raster, path))
