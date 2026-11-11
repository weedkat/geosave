"""Read-only xarray -> Item -> Arrow -> GeoDataFrame feasibility probe.

Run with the repository's Python environment. Does not read or write raster assets.
The encoding normalization below is an experimental policy, not a public API.
"""

from datetime import UTC, datetime
import json

import dask.array as da
from dask.callbacks import Callback
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pystac
from pystac.validation import JsonSchemaSTACValidator
from odc.geo.geobox import GeoBox
import xarray as xr
from xarray.conventions import encode_cf_variable

import geosave  # noqa: F401
from geosave.geodata.stac.item import create_item
from stac_geoparquet.arrow import parse_stac_items_to_arrow, stac_table_to_items


def make_raster(resolution=10):
    grid = GeoBox.from_bbox(
        (300000, 5000000, 300040, 5000040), "EPSG:32633", resolution=resolution
    )
    pixels = da.full(tuple(grid.shape), 0.2, chunks=(2, 2), dtype="float32")
    rows = xr.Dataset({"red": (("y", "x"), pixels)})
    return rows.gs.write_grid(grid).assign_coords(time=np.datetime64("2024-01-02"))


def encode_raster(raster):
    """Prototype: derive the stored representation using xarray CF encoding."""
    stored = raster.copy(deep=False)
    for name, variable in raster.data_vars.items():
        encoded = encode_cf_variable(variable.variable)
        stored[name] = xr.DataArray(encoded, coords=variable.coords, name=name)
    return stored


def check_item(item):
    # Validate JSON values: Shapely mapping uses tuples for coordinates.
    fields = json.loads(json.dumps(item.to_dict(transform_hrefs=False)))
    validator = JsonSchemaSTACValidator()
    validator.validate_core(fields, pystac.STACObjectType.ITEM, fields["stac_version"])

    arrow = parse_stac_items_to_arrow([item], drop_invalid_properties=False).read_all()
    frame = gpd.GeoDataFrame.from_arrow(
        arrow, to_pandas_kwargs={"types_mapper": pd.ArrowDtype}
    )
    restored = pa.table(frame.to_arrow(index=False))
    assert restored.equals(arrow, check_metadata=False)
    record = next(iter(stac_table_to_items(restored)))
    for name, asset in item.assets.items():
        assert record["assets"][name] == asset.to_dict()
    assert record["id"] == item.id
    assert frame.crs.to_epsg() == 4326
    return {"assets": list(item.assets), "rows": len(frame), "core_schema": "passed"}


def main():
    results = {}
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(1)):
        scene = make_raster()
        for suffix in ("tif", "nc", "zarr"):
            # Explicit URLs describe assets; no source URL is inferred from xarray.
            item = create_item(
                {"image": (f"https://example.com/scene.{suffix}", scene)},
                id_format=f"scene-{suffix}",
            )
            results[suffix] = check_item(item)
            results[suffix]["media_type"] = item.assets["image"].media_type

        tree = xr.DataTree.from_dict({"fine": scene, "coarse": make_raster(20)})
        assets = {}
        for name in ("fine", "coarse"):
            assets[name] = (f"https://example.com/{name}.tif", tree[name].to_dataset())
        item = create_item(assets, id_format="mixed-grid")
        results["mixed_grid_tree"] = check_item(item)
        shapes = []
        for asset in item.assets.values():
            shapes.append(asset.extra_fields["proj:shape"])
        assert shapes == [[4, 4], [2, 2]]
        results["mixed_grid_tree"]["shapes"] = shapes

        band = scene.red
        item = create_item(
            {"red": ("https://example.com/red.tif", band.to_dataset())},
            id_format="dataarray",
        )
        results["dataarray"] = check_item(item)

        composite = scene.gs.write_time_bounds(
            [(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC))]
        )
        item = create_item(
            {"image": ("https://example.com/composite.nc", composite)},
            id_format="composite",
        )
        results["composite"] = check_item(item)
        assert item.properties["start_datetime"].startswith("2024-01-01")
        assert item.properties["end_datetime"].startswith("2024-01-31")

        scene.red.encoding = {
            "dtype": "uint16",
            "scale_factor": 0.0001,
            "add_offset": 0.0,
            "_FillValue": 65535,
        }
        original_encoding = dict(scene.red.encoding)
        stored = encode_raster(scene)
        item = create_item(
            {"image": ("https://example.com/packed.nc", stored)},
            id_format="encoded",
        )
        results["encoded"] = check_item(item)
        fields = item.assets["image"].extra_fields["raster:bands"][0]
        assert fields["data_type"] == "uint16"
        assert fields["scale"] == 0.0001
        assert fields["nodata"] == 65535
        assert scene.red.dtype == np.dtype("float32")
        assert scene.red.encoding == original_encoding
        assert isinstance(stored.red.data, da.Array)
        results["encoded"]["band"] = fields

    assert tasks == []
    results["dask_tasks"] = len(tasks)
    results["asset_io"] = "No asset reader or writer invoked; URLs are illustrative."
    results["limits"] = [
        "Core STAC schemas validated; extension schemas not validated.",
        "No raster persistence or file round trip in this probe.",
        "CF encoding normalization is a prototype, not a writer-independent guarantee.",
        "DataTree asset mapping is explicit; no automatic collection/date grouping.",
        "gs.to_item is proposed; probe calls the existing create_item function.",
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
