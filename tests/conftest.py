from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array as da
import pytest
import xarray as xr
from dotenv import load_dotenv
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata.core.region import GeoRegion


def place(raster, geobox):
    """Put pixels on a grid as callers do with ODC: its coordinates and CRS coordinate."""
    return raster.assign_coords(xr_coords(geobox, dims=raster.odc.spatial_dims))


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
        raster = place(raster, reference)
    return raster if nodata is None else raster.gs.write_nodata(nodata)


def build_dataarray(pixels, reference=None, *, dims, nodata=None, **coords):
    """Build one band as callers do, through `build_dataset`."""
    raster = build_dataset(
        {"pixels": (dims, pixels)}, reference, nodata=nodata, **coords
    )
    return raster["pixels"].rename(None)


def write_cogs(raster, root, format_paths, **options):
    """Write a Dataset as COGs named by a path template, one per date, as test setup.

    A template naming ``{variable}`` writes a file per variable; otherwise every
    variable of a date shares one file.
    """
    import pandas as pd

    from geosave.geodata.io import storage
    from geosave.geodata.io.raster.geotiff import write_files

    folder = storage.create_upath(root, options.pop("storage_options", None))
    observations = [raster]
    if "time" in raster.dims:
        observations = []
        for position in range(raster.sizes["time"]):
            observations.append(raster.isel(time=position))
    files = {}
    for observation in observations:
        # Scalar coordinates fill the template, dates as Timestamps for strftime.
        labels = {}
        for name, coord in observation.coords.items():
            if coord.ndim != 0 or name == "spatial_ref":
                continue
            if coord.dtype.kind == "M":
                labels[str(name)] = pd.Timestamp(coord.values[()])
            else:
                labels[str(name)] = coord.item()
        if "{variable}" not in format_paths:
            files[folder / format_paths.format_map(labels)] = observation
            continue
        for name in observation.data_vars:
            path = format_paths.format_map({**labels, "variable": name})
            files[folder / path] = observation[[name]]
    return write_files(files, **options)


def combine_item(tree):
    """Combine the only Item of a tree loaded from one STAC Item."""
    import xarray as xr

    (path,) = {leaf.parent.path for leaf in tree.leaves}
    assets = []
    for leaf in tree[path].leaves:
        assets.append(leaf.to_dataset(inherit="all_coords"))
    return xr.merge(assets, join="exact")


def build_tree(rasters):
    """Group whole Datasets in a native tree."""
    return xr.DataTree.from_dict(rasters)


@pytest.fixture(autouse=True)
def headless_browser(monkeypatch):
    """Fail on desktop browser launches; browser tests use headless Playwright."""
    import webbrowser

    def refuse(*args, **kwargs):
        pytest.fail("Tests must not open the desktop browser; use headless Playwright")

    monkeypatch.setattr(webbrowser, "get", refuse)
    monkeypatch.setattr(webbrowser, "open", refuse)


def pytest_configure(config):
    # Keep catalogue namespaces distinct across task-specific recipe tests.
    # tests/cli exercises the flattened, single-task generated workspaces.
    from geosave.cli.core.templates import TASKS_DIR, get_methods, get_tasks
    from geosave.cli.core.copy import safe_copy
    from geosave.cli.core.workspace import create_workspace

    folder = TemporaryDirectory(prefix="geosave-test-recipes-")
    root = Path(folder.name)
    create_workspace(root)
    for task in get_tasks():
        for method in get_methods(task):
            safe_copy(
                TASKS_DIR / task / method, root, skip_all=True,
                exclude={"__pycache__", "description.txt"},
            )
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
