from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import dask.array as da
from odc.geo.geobox import GeoBox
import pytest
import pystac
from dotenv import load_dotenv

from geosave.geodata.core.anchor import GeoAnchor
from geosave.geodata.core.raster.create import raster
from geosave.geodata.stac.source import StacSource
from geosave.model.spec import ModelSpec, RasterRequirement


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
def spec():
    return ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(variables=("nir", "red"), require_crs=True)
        },
    )


@pytest.fixture
def raw():
    box = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    return {
        "optical": raster(
            {
                name: (("y", "x"), da.full((4, 4), value, chunks=(2, 2)))
                for name, value in (("red", 2.0), ("nir", 6.0), ("unused", 0.0))
            },
            box,
        )
    }


@pytest.fixture
def anchor(raw):
    return GeoAnchor(geobox=raw["optical"].gs.geobox)


@pytest.fixture
def source():
    # These unit tests replace loading; constructing a source performs no I/O.
    class EmptyCatalog:
        def search_items(self, query: object) -> list[pystac.Item]:
            return []

        def collection(self, collection: str) -> pystac.Collection:
            raise KeyError(collection)

    return StacSource(client=EmptyCatalog(), collection="example").set_config(
        bands=("unused",), chunks={"x": 2, "y": 2}, with_properties=("platform",)
    )


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
