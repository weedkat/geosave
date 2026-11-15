"""What `import geosave as gs` offers, and what it costs to import."""

from __future__ import annotations

import subprocess
import sys


def test_raster_tree_opener_is_available_at_each_public_entry_point() -> None:
    import geosave as gs
    from geosave import geodata
    from geosave.geodata import io

    assert gs.open_raster_tree is geodata.open_raster_tree is io.open_raster_tree


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
        [sys.executable, "-c", "import geosave.geodata.io.vector.stac_geoparquet"],
        check=True,
    )
