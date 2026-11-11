from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array as da
import pytest
import xarray as xr
from dotenv import load_dotenv
from odc.geo.geobox import GeoBox

from geosave.geodata.core.region import GeoRegion


def build_dataset(variables, reference=None, *, nodata=None, **coords):
    """Build a raster as callers do: xarray's constructor, then its grid and nodata.

    `reference` is a grid, or a raster lending its grid and its coordinates on
    the variables' other dimensions, such as `time`.
    """
    if isinstance(reference, (xr.Dataset, xr.DataArray)):
        lent = {}
        for name, coord in reference.coords.items():
            if name not in ("y", "x", "spatial_ref") and name not in coords:
                lent[name] = coord
        raster = xr.Dataset(variables, coords=coords)
        for name, coord in lent.items():
            if set(coord.dims) <= set(raster.dims):
                raster = raster.assign_coords({name: coord})
        reference = reference.gs.geobox
    else:
        raster = xr.Dataset(variables, coords=coords)
    if reference is not None:
        raster = raster.gs.write_grid(reference)
    return raster if nodata is None else raster.gs.write_nodata(nodata)


def build_dataarray(pixels, reference=None, *, dims, nodata=None, **coords):
    """Build one band as callers do, through `build_dataset`."""
    raster = build_dataset(
        {"pixels": (dims, pixels)}, reference, nodata=nodata, **coords
    )
    return raster["pixels"].rename(None)


def build_tree(rasters):
    """Merge rasters keyed by collection, or by `collection/res` path, into one tree."""
    from geosave.geodata.io.paths import format_resolution

    nodes = {}
    for key, raster in rasters.items():
        dataset = (
            raster.to_dataset(name=raster.name or key.rsplit("/", 1)[-1])
            if isinstance(raster, xr.DataArray)
            else raster
        )
        if "/" in key:
            collection = key.split("/", 1)[0]
            if collection not in nodes:
                nodes[collection] = xr.Dataset()
            nodes[key] = dataset
        else:
            nodes[key] = xr.Dataset(attrs=dict(dataset.attrs))
            node = dataset.copy()
            node.attrs = {}
            nodes[f"{key}/{format_resolution(dataset.gs.geobox)}"] = node
    return xr.DataTree.from_dict(nodes)


@pytest.fixture(autouse=True)
def headless_browser(monkeypatch):
    """Fail on desktop browser launches; browser tests use headless Playwright."""
    import webbrowser

    def refuse(*args, **kwargs):
        pytest.fail("Tests must not open the desktop browser; use headless Playwright")

    monkeypatch.setattr(webbrowser, "get", refuse)
    monkeypatch.setattr(webbrowser, "open", refuse)


def pytest_configure(config):
    # Train recipe tests against copied project code, as consumers use it.
    from geosave.cli.core.templates import get_methods, get_tasks
    from geosave.cli.core.workspace import add_task, create_workspace

    folder = TemporaryDirectory(prefix="geosave-test-recipes-")
    root = Path(folder.name)
    create_workspace(root)
    for task in get_tasks():
        for method in get_methods(task):
            add_task(root, task, method)
    sys.path.insert(0, str(root))
    config.add_cleanup(lambda: sys.path.remove(str(root)))
    config.add_cleanup(folder.cleanup)
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


@pytest.fixture
def raw():
    box = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    return {
        "optical": build_dataset(
            {
                name: (("y", "x"), da.full((4, 4), value, chunks=(2, 2)))
                for name, value in (("red", 2.0), ("nir", 6.0), ("unused", 0.0))
            },
            box,
        )
    }


@pytest.fixture
def grid(raw):
    return raw["optical"].gs.geobox


@pytest.fixture
def region(raw):
    return GeoRegion(raw["optical"].gs.geobox.extent)


@pytest.fixture
def bucket():
    """Yield a unique `memory://` prefix and empty it afterwards."""
    import uuid

    import fsspec

    filesystem = fsspec.filesystem("memory")
    prefix = f"memory://geosave-tests/{uuid.uuid4()}"
    yield prefix
    if filesystem.exists(prefix):
        filesystem.rm(prefix, recursive=True)
