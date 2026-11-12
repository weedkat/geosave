"""What `import geosave as gs` offers, and what it costs to import."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_root_registers_raster_and_vector_accessors_without_torch() -> (
    None
):
    code = (
        "import sys, geosave, geopandas as gpd, xarray as xr; "
        "owners = (xr.Dataset, xr.DataArray, xr.DataTree, gpd.GeoDataFrame); "
        "missing = [o.__name__ for o in owners if 'gs' not in dir(o)]; "
        "assert not missing, missing; "
        "heavy = [m for m in ('torch', 'lightning', 'holoviews', 'panel') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_stac_geoparquet_can_load_before_stac() -> None:
    subprocess.run(
        [sys.executable, "-c", "import geosave.geodata.io.formats.stac_geoparquet"],
        check=True,
    )
