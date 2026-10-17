"""Runnable native explorer proof, independent of remote data services.

    panel serve scripts/smoke_explorer.py

In a notebook, import `create_explorer` and display its return value. Set
GEOSAVE_PROOF_SIZE=8192 for the larger raster-work measurement fixture.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import weakref

import dask.array as da
import geopandas as gpd
import panel as pn
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from shapely.geometry import Point, box

import geosave  # noqa: F401
from geosave.geodata import stac
from geosave.geodata.viz.explorer import Explorer, RasterLayer


def create_explorer() -> Explorer:
    """Build a selectable COG catalog, vector and computed lazy RGB raster."""
    directory = TemporaryDirectory(prefix="geosave-holoviz-")
    size = int(os.environ.get("GEOSAVE_PROOF_SIZE", "2048"))
    grid = GeoBox.from_bbox(
        (-1000000, 5000000, -900000, 5100000), crs=3857, shape=(size, size), tight=True
    )
    indices = da.indices((size, size), chunks=(256, 256))
    y, x = indices[0], indices[1]
    raster = xr.Dataset(
        {
            "red": xr.DataArray(
                (da.sin(x / 80) + 1).astype("float32"),
                dims=("y", "x"),
                coords=xr_coords(grid),
            ),
            "green": xr.DataArray(
                (da.cos(y / 90) + 1).astype("float32"),
                dims=("y", "x"),
                coords=xr_coords(grid),
            ),
            "blue": xr.DataArray(
                ((x + y) / size).astype("float32"),
                dims=("y", "x"),
                coords=xr_coords(grid),
            ),
        }
    ).gs.write_rgb("red", "green", "blue")
    # File creation is explicit fixture work, outside display measurements.
    files = raster.gs.to_cog(Path(directory.name) / "image.tif")
    catalog = stac.create_catalog(
        stac.create_items(
            files, id_format="COG", datetime=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
    )
    catalog["quality"] = 90
    fields = gpd.GeoDataFrame(
        {"crop": ["rice", "palm"], "area": [2.0, 5.0]},
        geometry=[box(-990000, 5010000, -970000, 5030000), Point(-940000, 5060000)],
        crs=3857,
    )
    viewer = Explorer(
        catalog,
        fields,
        RasterLayer(raster, name="Lazy xarray"),
        basemap="None",
        height=430,
    )
    cover = (((x // 128) + (y // 128)) % 3).astype("uint8")
    labels = xr.DataArray(cover, dims=("y", "x"), coords=xr_coords(grid), name="cover")
    labels = labels.gs.write_legend(
        {0: "Water", 1: "Trees", 2: "Crops"},
        {0: "#419bdf", 1: "#397d49", 2: "#e49635"},
    )
    classified = RasterLayer(labels, name="Land cover", alpha=0.8)
    classified.visible = False
    viewer.add(classified)
    weakref.finalize(viewer, directory.cleanup)
    if pn.state.served:
        pn.state.on_session_destroyed(lambda context: directory.cleanup())
    return viewer


if pn.state.served:
    create_explorer().servable()
