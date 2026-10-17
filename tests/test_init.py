"""What `import geosave as gs` offers, and what it costs to import."""

from __future__ import annotations

import subprocess
import sys

import pytest

import geosave as gs


@pytest.mark.parametrize("name", ["features", "stac"])
def test_the_root_offers_the_geodata_modules(name: str) -> None:
    from geosave import geodata

    assert getattr(gs, name) is getattr(geodata, name)


def test_the_root_array_constructor_builds_native_xarray() -> None:
    import numpy as np
    import xarray as xr

    from geosave import geodata

    pixels = np.zeros((2, 3), dtype="uint8")
    built = gs.array(pixels, dims=("y", "x"))

    assert gs.array is geodata.array
    assert isinstance(built, xr.DataArray)
    assert built.name is None
    assert built.data is pixels


def test_importing_the_root_registers_raster_and_vector_accessors_without_torch() -> (
    None
):
    code = (
        "import sys, geosave, geopandas as gpd, xarray as xr; "
        "owners = (xr.Dataset, xr.DataArray, xr.DataTree, gpd.GeoDataFrame); "
        "missing = [o.__name__ for o in owners if 'gs' not in dir(o)]; "
        "assert not missing, missing; "
        "loaded = [m for m in ('windows', 'features', 'stac') "
        "if f'geosave.geodata.{m}' in sys.modules]; "
        "assert not loaded, loaded; "
        "heavy = [m for m in ('torch', 'lightning', 'holoviews', 'panel') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_root_offers_the_attrs_groups() -> None:
    from geosave.geodata import attrs

    assert gs.attrs is attrs
    assert gs.attrs.ACDD(id="label", summary="Classes").id == "label"


@pytest.mark.parametrize(
    "name", ["tree", "open_tree", "GeoDataTree", "GeoDataset", "GeoDataArray"]
)
def test_the_root_names_objects_after_xarray(name: str) -> None:
    assert hasattr(gs, name)


@pytest.mark.parametrize(
    "name", ["stack", "open_stack", "GeoStack", "GeoRaster", "GeoArray"]
)
def test_the_root_drops_the_old_vocabulary(name: str) -> None:
    assert not hasattr(gs, name)


@pytest.mark.parametrize("module", ["stac_geoparquet", "stac_json"])
def test_vector_formats_can_load_before_stac(module: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"import geosave.geodata.io.vector.{module}"],
        check=True,
    )
