"""Run with .venv/bin/python docs/contributing/benchmarks/2026-11-06-native-open/feasibility.py."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array as da
import numpy as np
import pystac
import rasterio
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave.geodata import io
from geosave.geodata.io import open_item
from geosave.geodata.stac.item import TIMELESS


def make_item(folder, identity, grids):
    entry = pystac.Item(identity, None, None, datetime(2025, 1, 15, tzinfo=UTC), {})
    entry.collection_id = "local"
    entry.stac_extensions = [
        "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
    ]
    footprint = None
    for index, (name, grid) in enumerate(grids.items(), start=1):
        path = folder / f"{identity}-{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=1,
            dtype="uint16",
            crs=str(grid.crs),
            transform=grid.transform,
            nodata=65535,
        ) as file:
            file.write(np.full(tuple(grid.shape), index, dtype="uint16"), 1)
        entry.add_asset(
            name,
            pystac.Asset(
                str(path),
                media_type=pystac.MediaType.COG,
                roles=["data"],
                extra_fields={
                    "proj:code": str(grid.crs),
                    "proj:shape": list(grid.shape),
                    "proj:transform": list(grid.transform)[:6],
                },
            ),
        )
        polygon = grid.extent.to_crs("EPSG:4326")
        footprint = polygon if footprint is None else footprint.union(polygon)
    entry.geometry = footprint.json
    entry.bbox = list(footprint.boundingbox)
    return entry


with TemporaryDirectory() as temporary:
    folder = Path(temporary)
    grid = GeoBox((4, 4), Affine(10, 0, 500000, 0, -10, 9000000), "EPSG:32749")
    shifted = GeoBox((4, 4), Affine.translation(5, 0) * grid.transform, grid.crs)
    entry = make_item(folder, "aligned", {"red": grid})
    with open_item(entry) as native, io.load_dataset([entry], geobox=grid) as loaded:
        raster = next(iter(native.gs.datasets.values()))
        assert isinstance(raster.red.data, da.Array)
        assert isinstance(loaded.red.data, da.Array)
        np.testing.assert_array_equal(raster.red.values, loaded.red.isel(time=0).values)
        assert raster.gs.geobox == loaded.gs.geobox
        print(
            f"aligned COG: equal pixels/grid and lazy arrays; dtype {raster.red.dtype} -> {loaded.red.dtype}; scalar time -> time dimension"
        )

    entry = make_item(folder, "shifted", {"red": grid, "nir": shifted})
    with open_item(entry) as native, io.load_datatree([entry]) as loaded:
        grids = {}
        for raster in native.gs.datasets.values():
            for name in raster.data_vars:
                grids[name] = raster.gs.geobox
        assert grids["red"] == grid and grids["nir"] == shifted
        assert len(native.gs.datasets) == 2
        assert len(loaded.gs.datasets) == 1
        print("shifted same-resolution assets: native retains 2 grids; ODC produces 1")

    entry = make_item(folder, "interval", {"red": grid})
    entry.properties.update(
        start_datetime="2025-01-01T00:00:00Z", end_datetime="2025-01-31T23:59:59Z"
    )
    with open_item(entry) as native, io.load_dataset([entry], geobox=grid) as loaded:
        raster = next(iter(native.gs.datasets.values()))
        assert "start_datetime" in raster.coords
        assert "start_datetime" not in loaded.coords
        print(
            "composite time bounds: native retains coordinates; ODC output omits them"
        )

    entry = make_item(folder, "timeless", {"red": grid})
    entry.datetime = None
    entry.properties.update(
        start_datetime=TIMELESS[0].isoformat(), end_datetime=TIMELESS[1].isoformat()
    )
    with open_item(entry) as native:
        assert "time" not in next(iter(native.gs.datasets.values())).dims
    try:
        with io.load_dataset([entry], geobox=grid) as loaded:
            assert "time" in loaded.dims
            print("timeless: native has no time axis; ODC constructs a time axis")
    except (ValueError, OverflowError) as error:
        print(
            f"timeless: native opens without time; ODC rejects: {type(error).__name__}: {error}"
        )
