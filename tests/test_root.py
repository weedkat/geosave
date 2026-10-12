"""What `import geosave_engine as gs` offers, and what it costs to import."""

from __future__ import annotations

import subprocess
import sys

import pytest

import geosave_engine as gs


@pytest.mark.parametrize("name", ["cuts", "features", "stac"])
def test_the_root_offers_the_geodata_modules(name: str) -> None:
    from geosave_engine import geodata

    assert getattr(gs, name) is getattr(geodata, name)


def test_importing_the_root_loads_none_of_them() -> None:
    code = (
        "import sys, geosave_engine; "
        "loaded = [m for m in ('cuts', 'features', 'stac') "
        "if f'geosave_engine.geodata.{m}' in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_root_array_constructor_builds_native_xarray() -> None:
    import numpy as np
    import xarray as xr

    from geosave_engine import geodata

    pixels = np.zeros((2, 3), dtype="uint8")
    built = gs.array(pixels, dims=("y", "x"))

    assert gs.array is geodata.array
    assert isinstance(built, xr.DataArray)
    assert built.name is None
    assert built.data is pixels


def test_importing_the_root_registers_raster_and_vector_accessors_without_torch() -> None:
    code = (
        "import sys, geosave_engine, geopandas as gpd, xarray as xr; "
        "owners = (xr.Dataset, xr.DataArray, xr.DataTree, gpd.GeoDataFrame); "
        "missing = [o.__name__ for o in owners if 'gs' not in dir(o)]; "
        "assert not missing, missing; "
        "heavy = [m for m in ('torch', 'lightning', 'holoviews', 'panel') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
