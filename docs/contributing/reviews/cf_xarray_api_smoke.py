"""Print observed cf-xarray 0.11.3 API limits; not desired-behavior tests.

Run from the repository root with .venv/bin/python and MPLCONFIGDIR=/tmp/geosave-mpl.
No files or library objects outside these small examples are modified.
"""

import warnings
import numpy as np
import xarray as xr
import cf_xarray  # noqa: F401 registers .cf
from dask.callbacks import Callback
from pyproj import CRS

x = xr.Dataset(
    {
        "image": (
            ("row", "col"),
            np.arange(6.0).reshape(2, 3),
            {"standard_name": "air_temperature", "cell_measures": "area: area"},
        ),
        "area": (("row", "col"), np.ones((2, 3))),
    },
    coords={
        "row": ("row", [0.0, 1.0], {"axis": "Y"}),
        "col": ("col", [0.0, 1.0, 2.0], {"axis": "X"}),
    },
).chunk({"row": 1, "col": 3})
tasks = []
with Callback(pretask=lambda *args: tasks.append(args)):
    reduced = x.cf["air_temperature"].cf.mean("X")
    selected = x.cf.sel(X=slice(0, 1))
    rolling = x.cf["air_temperature"].cf.rolling(X=2).mean()
    weighted = x.cf["air_temperature"].cf.weighted("area").mean(("X", "Y"))
print(
    "DYNAMIC wrapper graph tasks",
    len(tasks),
    "mean dims",
    reduced.dims,
    "selected sizes",
    dict(selected.sizes),
)
print("WEIGHTED result", weighted.compute().item())
print("ROLLING", rolling.compute().values.tolist())
print("SIZES mapping", x.cf.sizes)
mapping = {"pixel": ("Y", "X")}
x.cf.stack(mapping)
print("STACK mapping afterward", mapping)
x.image.attrs["ancillary_variables"] = "missing"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    relations = x.cf.get_associated_variable_names("image")
print(
    "MISSING references retained",
    relations["ancillary_variables"],
    "warnings",
    len(caught),
)
try:
    x.cf.drop("area")
except Exception as e:
    print("DROP", type(e).__name__)
for method in ["isin", "plot"]:
    try:
        if method == "isin":
            x.cf.isin(["cloud"])
        else:
            x.cf.plot
    except Exception as e:
        print("DATASET", method, type(e).__name__)
# Public helper contract on attrs vs decoded encoding.
b = xr.Dataset(
    {"edges": (("time", "nv"), [[0, 1], [1, 2]])}, coords={"time": [0.5, 1.5]}
)
b.time.encoding["bounds"] = "edges"
print("BOUND selection", b.cf.get_bounds("time").name)
try:
    b.cf.get_bounds_dim_name("time")
except Exception as e:
    print("BOUND dimension lookup", type(e).__name__)


image = xr.DataArray(
    [[1.0, 2.0], [3.0, 4.0]],
    dims=("lat", "lon"),
    coords={
        "lat": (
            "lat",
            [0.0, 1.0],
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        "lon": (
            "lon",
            [0.0, 1.0],
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
        "crs": xr.DataArray(0, attrs=CRS.from_epsg(4326).to_cf()),
    },
    name="image",
)
for location in ("attrs", "encoding"):
    image.attrs.clear()
    image.encoding.clear()
    getattr(image, location)["grid_mapping"] = "crs"
    try:
        print(location, "grid mappings", len(image.cf.grid_mappings))
    except AttributeError as error:
        print(location, "grid mappings", type(error).__name__, str(error))
    print(location, "discovered keys", sorted(image.cf.keys()))
