"""Print each part of a GeoSave raster on disk, in xarray, through .cf and through .gs.

Run with `uv run python docs/contributing/reviews/cf_structure_reference.py`.
The output backs `2026-11-11-cf-structure-reference.md`.
"""

import json
import tempfile
import warnings
from pathlib import Path

import dask.array as da
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geom import box

import geosave  # noqa: F401 registers .gs
import geosave.geodata.attrs as attrs

warnings.simplefilter("ignore")


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def show(label, value):
    print(f"--- {label}")
    print(value)


# ---------------------------------------------------------------- target object
times = pd.to_datetime(["2025-01-10T10:00", "2025-01-20T10:00"]).values
cells = np.stack(
    [pd.to_datetime(["2025-01-01", "2025-01-15"]), pd.to_datetime(["2025-01-15", "2025-02-01"])],
    axis=1,
).astype("datetime64[ns]")
y = [5000015.0, 5000005.0]
x = [500005.0, 500015.0, 500025.0]


def pixels(dtype):
    return da.ones((2, 2, 3), dtype=dtype, chunks=(1, 2, 3))


scene = xr.Dataset(
    {
        "red": (("time", "y", "x"), pixels("uint16"), {"long_name": "Red reflectance", "standard_name": "surface_bidirectional_reflectance", "units": "1", "cell_methods": "time: mean"}),
        "nir": (("time", "y", "x"), pixels("uint16"), {"units": "1"}),
        "scl": (("time", "y", "x"), pixels("uint8"), {"long_name": "Scene classification"}),
        "dem": (("y", "x"), da.zeros((2, 3), dtype="float32", chunks=(2, 3)), {"units": "m", "standard_name": "height_above_reference_ellipsoid"}),
    },
    coords={"time": ("time", times, {"standard_name": "time", "axis": "T"}), "y": y, "x": x},
)
# CF cell bounds: a variable named by the time coordinate's `bounds` attr.
scene = scene.assign_coords(time_bounds=(("time", "bounds"), cells))
scene.time.attrs["bounds"] = "time_bounds"

# Everything else through GeoSave's own writers.
scene = scene.gs.write_crs("EPSG:32633")
scene = scene.gs.write_scale_offset(1e-4, target=["red", "nir"])
scene = scene.gs.write_nodata(0, target=["red", "nir"])
scene = scene.gs.write_spectral("red", center_wavelength=0.665, target="red")
scene = scene.gs.write_spectral("nir", center_wavelength=0.842, target="nir")
scene = scene.gs.write_legend({0: "no_data", 8: "cloud_medium", 9: "cloud_high"}, {9: "#FFFFFF"}, target="scl")
scene = scene.gs.write_acdd(attrs.ACDD(id="sentinel-2-l2a", title="Sentinel-2 L2A"))
scene = scene.gs.write_extent(box(500000, 5000000, 500030, 5000020, "EPSG:32633"))
records = [{"platform": "sentinel-2a", "eo:cloud_cover": 10.0}, {"platform": "sentinel-2b", "eo:cloud_cover": 40.0}]
scene = scene.assign_coords(stac_properties=("time", [json.dumps(r) for r in records]))
scene = attrs.stamp_attrs(scene, attrs.StacMetadata(collection={"id": "sentinel-2-l2a"}, extensions=["https://stac-extensions.github.io/eo/v1.1.0/schema.json"]))

section("xarray in memory, as built")
print(scene)

# ---------------------------------------------------------------- on disk
folder = Path(tempfile.mkdtemp())
from geosave.geodata.io.formats import netcdf as gs_netcdf, zarr as gs_zarr  # noqa: E402

# Written by GeoSave's own writers, as GeoSave stores a raster.
gs_netcdf.write_netcdf(scene, folder / "scene.nc")
gs_zarr.write_zarr(scene, folder / "scene.zarr")

section("on disk: netCDF variables (raw netCDF4, no xarray)")
with netCDF4.Dataset(folder / "scene.nc") as raw:
    show("global attrs", {k: raw.getncattr(k) for k in raw.ncattrs()})
    for name, variable in raw.variables.items():
        print(f"{name}{variable.dimensions} {variable.dtype}: {dict((k, variable.getncattr(k)) for k in variable.ncattrs())}")

section("on disk: Zarr v3 arrays (raw zarr.json attributes)")
for meta in sorted((folder / "scene.zarr").glob("*/zarr.json")):
    body = json.loads(meta.read_text())
    print(f"{meta.parent.name}{tuple(body.get('dimension_names') or ())} {body['data_type']} fill={body.get('fill_value')}: {body['attributes']}")
root = json.loads((folder / "scene.zarr" / "zarr.json").read_text())
show("root attributes", root["attributes"])

# ---------------------------------------------------------------- reopened
opened = {
    "plain xr.open_dataset (xarray defaults)": xr.open_dataset(folder / "scene.nc", chunks={}),
    "GeoSave open_netcdf (defaults)": gs_netcdf.open_netcdf(folder / "scene.nc", chunks={}),
    "GeoSave open_netcdf(mask_and_scale=True)": gs_netcdf.open_netcdf(folder / "scene.nc", chunks={}, mask_and_scale=True),
}

for label, ds in opened.items():
    if ds is None:
        continue
    section(f"reopened: {label}")
    show("coords / data_vars", (sorted(map(str, ds.coords)), sorted(map(str, ds.data_vars))))
    show("time.attrs / time.encoding[bounds, units, calendar]", (ds.time.attrs, {k: ds.time.encoding.get(k) for k in ("bounds", "units", "calendar")}))
    show("red dtype / attrs", (ds.red.dtype, ds.red.attrs))
    show("red.encoding[_FillValue, scale_factor, add_offset, dtype, grid_mapping, coordinates]", {k: ds.red.encoding.get(k) for k in ("_FillValue", "scale_factor", "add_offset", "dtype", "grid_mapping", "coordinates")})
    show("scl.attrs", ds.scl.attrs)
    show("spatial_ref attrs keys", sorted(ds.spatial_ref.attrs))
    show("root attrs keys", sorted(ds.attrs))
    show(".cf.bounds", ds.cf.bounds)
    show(".cf.axes / .cf.coordinates", (ds.cf.axes, ds.cf.coordinates))
    show(".cf.standard_names", ds.cf.standard_names)
    show(".cf.grid_mapping_names", ds.cf.grid_mapping_names)
    show(".gs.packing['red'] / .gs.nodata['red']", (ds.gs.packing["red"], ds.gs.nodata["red"]))
    show(".gs.cell_bounds dims", None if ds.gs.cell_bounds is None else ds.gs.cell_bounds.dims)

zopened = gs_zarr.open_zarr(folder / "scene.zarr")
section("reopened: GeoSave open_zarr (defaults)")
show("coords / data_vars", (sorted(map(str, zopened.coords)), sorted(map(str, zopened.data_vars))))
show("time.attrs / time.encoding[bounds]", (zopened.time.attrs, zopened.time.encoding.get("bounds")))
show("red dtype / attrs", (zopened.red.dtype, zopened.red.attrs))
show("scl.attrs", zopened.scl.attrs)

# ---------------------------------------------------------------- views per piece
section(".cf and .gs on the in-memory scene")
show("cf.get_bounds('time')", scene.cf.get_bounds("time").values.astype("datetime64[D]").tolist())
show("cf.grid_mappings[0].name / .crs", (scene.cf.grid_mappings[0].name, scene.cf.grid_mappings[0].crs.to_epsg()))
show("scl.cf.flags", scene.scl.cf.flags)
show("(scl.cf == 'cloud_high') is lazy", type((scene.scl.cf == "cloud_high").data).__name__)
show("gs.time_coordinate.name / gs.times", (scene.gs.time_coordinate.name, list(scene.gs.times.strftime("%Y-%m-%d"))))
show("gs.time_bounds (start/end coords)", scene.gs.time_bounds)
show("gs.timespan", scene.gs.timespan)
show("gs.crs_name / gs.spatial_dims / gs.grid_mapping_coords", (scene.gs.crs_name, scene.gs.spatial_dims, scene.gs.grid_mapping_coords))
show("gs.support_variables / gs.raster_variables", (scene.gs.support_variables, scene.gs.raster_variables))
show("gs.packing['red']", scene.gs.packing["red"])
show("gs.legend['scl']", scene.gs.legend["scl"])
show("gs.spectral['red']", scene.gs.spectral["red"])
show("Quantity.from_attrs(red.attrs)", attrs.Quantity.from_attrs(scene.red.attrs))
show("gs.acdd", scene.gs.acdd)
show("Extent.from_attrs(scene.attrs)", attrs.Extent.from_attrs(scene.attrs))
show("Extent.from_attrs(red.attrs)", attrs.Extent.from_attrs(scene.red.attrs))
show("StacMetadata.from_attrs(scene.attrs)", attrs.StacMetadata.from_attrs(scene.attrs))
show("gs.stac_common_properties", scene.gs.stac_common_properties)

section("DataArray and selection views")
red = scene["red"]
show("scene['red'] coords / time.attrs", (sorted(map(str, red.coords)), red.time.attrs))
show("scene['red'].attrs keys", sorted(red.attrs))
show("scene[['red']] variables", sorted(map(str, scene[["red"]].variables)))
show("scene.cf[['red']] variables", sorted(map(str, scene.cf[["red"]].variables)))
show("scene.isel(time=0).time_bounds dims", scene.isel(time=0).time_bounds.dims)

section("DataTree: time and bounds on the parent node")
parent = xr.Dataset(coords={"time": scene.time, "time_bounds": scene.time_bounds, "stac_properties": scene.stac_properties}, attrs=dict(scene.attrs))
child = scene[["red", "nir"]].drop_vars(["stac_properties"]).drop_attrs(deep=False)
tree = xr.DataTree.from_dict({"/": parent, "/r10m": child})
node = tree["r10m"]
show("node.to_dataset() coords", sorted(map(str, node.to_dataset().coords)))
show("node.to_dataset(inherit='all_coords') coords", sorted(map(str, node.to_dataset(inherit="all_coords").coords)))
show("node.attrs keys (root attrs not inherited)", sorted(node.attrs))
show("tree.gs.datasets['r10m'] coords", sorted(map(str, tree.gs.datasets["r10m"].coords)) if "r10m" in tree.gs.datasets else list(tree.gs.datasets))
