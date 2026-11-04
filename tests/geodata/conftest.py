from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.geodata import configure_gdal
from geosave.geodata.stac.catalog import from_items
from tests.conftest import build_dataset as build
from tests.conftest import build_tree
from geosave.geodata.io import storage


def flat(paths):
    """Flatten a native raster/group path inventory in its declared order."""
    from collections.abc import Mapping
    from os import PathLike
    from upath import UPath

    if isinstance(paths, (str, PathLike, UPath)):
        return [paths]
    values = paths.values() if isinstance(paths, Mapping) else paths
    result = []
    for value in values:
        result.extend(flat(value))
    return result


def create_file_items(
    files,
    *,
    id_format="{start:%Y%m%dT%H%M%S}_{lat:.4f}_{lon:.4f}",
    collection=None,
    datetime=None,
):
    """Yield the Items of files already written, describing each file where it is.

    `files` is a path, a sequence of paths grouped into one Item per date, or
    asset keys mapped to paths of one Item.
    """
    from collections.abc import Mapping
    from os import PathLike

    import pystac
    from upath import UPath

    from geosave.geodata import open_raster
    from geosave.geodata.io.storage import location
    from geosave.geodata.stac.item import create_item
    from geosave.geodata.io.stac import group_files_by_date

    name = collection.id if isinstance(collection, pystac.Collection) else collection
    if isinstance(files, Mapping):
        observations = [
            {key: (location(path), open_raster(path)) for key, path in files.items()}
        ]
    else:
        paths = [files] if isinstance(files, (str, PathLike, UPath)) else list(files)
        opened = {location(path): open_raster(path) for path in paths}
        observations = group_files_by_date(opened)
    for observation in observations:
        yield create_item(
            observation, id_format=id_format, collection=name, datetime=datetime
        )


def create_test_collection(collection_id, *, description, **fields):
    """Return a Collection spanning the globe and all time, for fixture tables."""
    import pystac

    extent = pystac.Extent(
        pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
        pystac.TemporalExtent([[None, None]]),
    )
    return pystac.Collection(collection_id, description, extent, **fields)


def item_table(paths, *, id=None, prefix=None, **options):
    """Describe native writer results with fixture-specific Item ids."""
    from collections.abc import Mapping
    from datetime import UTC
    from pathlib import PurePath
    from geosave.geodata import open_raster

    spans = []
    for path in flat(paths):
        with open_raster(path) as raster:
            if raster.gs.timespan is not None:
                spans.append(raster.gs.timespan)
    if spans and "datetime" not in options:
        start = min(start for start, _ in spans).replace(tzinfo=UTC)
        end = max(end for _, end in spans).replace(tzinfo=UTC)
        options["datetime"] = (start, end)
    selected = options.pop("collection", None)
    groups = paths if isinstance(paths, Mapping) else {selected: paths}
    items = []
    for collection, files in groups.items():
        for item in create_file_items(
            files, collection=collection, id_format=id or "fixture", **options
        ):
            name = (
                item.id if id else PurePath(next(iter(item.assets.values())).href).stem
            )
            item.id = name if prefix is None else f"{prefix}/{name}"
            items.append(item)
    return from_items(
        items,
        collections=[
            create_test_collection(key, description=key)
            for key in dict.fromkeys(item.collection_id for item in items)
            if key is not None
        ],
    )


def build_raster(
    *,
    crs: str = "EPSG:32749",
    times: int = 0,
    packed: bool = False,
) -> xr.Dataset:
    """Build one raster the way an adapter would.

    Args:
        crs: CRS of the two-by-two grid.
        times: Length of the time axis, or 0 for a timeless raster.
        packed: Give each variable CF packing over stored digital numbers.

    Returns:
        Raster Dataset holding `red` and `nir` as uint16.

    Examples:
        >>> build_raster(times=2).red.dims
        ('time', 'y', 'x')
    """
    geobox = GeoBox.from_bbox(
        (500_000.0, 9_000_000.0, 500_020.0, 9_000_020.0),
        crs=crs,
        shape=(2, 2),
        tight=True,
    )
    coords = dict(xr_coords(geobox, always_yx=True))
    spatial_dims = ("y", "x")
    values = np.array([[1000, 2000], [3000, 0]], dtype="uint16")
    if times:
        coords["time"] = xr.DataArray(
            np.array(
                [f"2025-06-{day + 1:02d}" for day in range(times)],
                dtype="datetime64[ns]",
            ),
            dims="time",
        )
        block = np.broadcast_to(values, (times, 2, 2)).copy()
        dims = ("time", *spatial_dims)
    else:
        block = values
        dims = spatial_dims

    raw = xr.Dataset({"red": (dims, block), "nir": (dims, block)}, coords=coords)
    if packed:
        for name in raw.data_vars:
            raw[name].attrs.update(
                {
                    "scale_factor": np.float32(1e-4),
                    "add_offset": np.float32(0.0),
                    "_FillValue": np.uint16(0),
                }
            )
    return raw.gs.write_crs()


@pytest.fixture
def raster() -> xr.Dataset:
    return build_raster()


@pytest.fixture
def tree(raster: xr.Dataset) -> xr.DataTree:
    return build_tree({"optical": raster[["red"]], "infrared": raster[["nir"]]})


@pytest.fixture
def grid() -> GeoBox:
    return GeoBox.from_bbox(
        (300000, 5000000, 300640, 5000640), "EPSG:32633", resolution=10
    )


@pytest.fixture
def scene(grid):
    """Two dates of two uint16 bands with a fill value."""
    times = pd.to_datetime(["2025-06-01", "2025-06-11"])
    cube = np.arange(2 * grid.height * grid.width, dtype="uint16").reshape(
        2, *grid.shape
    )
    data = build(
        {"red": (("time", "y", "x"), cube), "nir": (("time", "y", "x"), cube + 1)},
        grid,
        time=times,
    )
    for name in data.data_vars:
        data[name].attrs["_FillValue"] = np.uint16(0)
    return data


@pytest.fixture
def label(grid):
    """One timeless uint8 class plane on the same grid."""
    return build({"label": (("y", "x"), np.ones(grid.shape, dtype="uint8"))}, grid)


def cog_template(name: str, *, dated: bool) -> str:
    """Spell, as a `to_cog` template, the layout one raster's files are expected in.

    A dated raster is a folder named after it holding one file per instant; an
    undated one is one file.
    """
    if dated:
        return f"{name}/{name}_{{time:%Y%m%dT%H%M%S}}.tif"
    return f"{name}.tif"


def tree_templates(sample) -> dict[str, str]:
    """Give every group of a tree its `cog_template` relative to the destination."""
    templates = {}
    for name in sample.gs.groups:
        raster = sample[name].gs.to_dataset()
        templates[name] = cog_template(name, dated="time" in raster.dims)
    return templates


def write_collection_cogs(tree, root, templates=None) -> dict[str, list]:
    """Write each collection of a tree as COGs, keyed by collection, as `item_table` takes them."""
    templates = templates or tree_templates(tree)
    files = {}
    for name in tree.gs.groups:
        files[name] = (
            tree[name].gs.to_dataset().gs.to_cog(root, path_format=templates[name])
        )
    return files


def read_items(rows):
    """Read one collection's rows as one raster: each a scene, joined along time."""
    from geosave.geodata.io.stac import open_scene
    from geosave.geodata.transform.temporal import concat

    scenes = []
    for entry in rows.gs.to_items():
        scenes.append(open_scene(entry).gs.to_dataset())
    return concat(scenes) if len(scenes) > 1 else scenes[0]


def titled(obj, title: str = "scene"):
    """Return a copy of a raster or tree whose root names its title."""
    copied = obj.copy()
    copied.attrs = {**obj.attrs, "title": title}
    return copied


def geobox(*, crs: str = "EPSG:32633", shape: tuple[int, int] = (8, 8)) -> GeoBox:
    """Build one grid for a raster-construction test.

    Args:
        crs: CRS the grid sits in.
        shape: Grid height and width in pixels.

    Returns:
        Grid of `shape` at ten-unit pixels.
    """
    left, bottom = 300_000.0, 5_000_000.0
    return GeoBox.from_bbox(
        (left, bottom, left + shape[1] * 10, bottom + shape[0] * 10),
        crs=crs,
        shape=shape,
        tight=True,
    )


@pytest.fixture
def rows(scene, label, tmp_path):
    sample = build_tree({"optical": scene, "label": label})
    saved = write_collection_cogs(sample, tmp_path / "s0")
    return item_table(saved, prefix="s0")


@pytest.fixture
def torch():
    """Load torch only for tests that convert pixels to tensors."""
    return pytest.importorskip("torch")


_GDAL_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "AWS_S3_ENDPOINT",
    "AWS_VIRTUAL_HOSTING",
    "AWS_HTTPS",
    "GDAL_DISABLE_READDIR_ON_OPEN",
)


def _refuse(*args, **kwargs):
    raise AssertionError("pixels were computed")


def _size(url: str, options: dict) -> int:
    return storage.location(url, options).stat().st_size


def _s3_service() -> tuple[str, dict, str]:
    """Return the prefix, storage options and GDAL endpoint of an S3 service."""
    endpoint = os.environ["GEOSAVE_S3_ENDPOINT"]
    bucket = os.environ.get("GEOSAVE_S3_BUCKET", "predictions")
    options = {
        "key": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        "client_kwargs": {"endpoint_url": endpoint},
    }
    storage.location(f"s3://{bucket}", options).mkdir(exist_ok=True)
    return f"s3://{bucket}", options, endpoint


def _hf_bucket() -> tuple[str, dict, str]:
    """Return the prefix, storage options and GDAL endpoint of an HF bucket."""
    return f"hf://buckets/{os.environ['GEOSAVE_HF_BUCKET']}", {}, "https://s3.hf.co"


@pytest.fixture(params=["s3", "hf"])
def remote(request):
    """Yield a unique remote prefix and its storage options, emptied afterwards."""
    variable = {"s3": "GEOSAVE_S3_ENDPOINT", "hf": "GEOSAVE_HF_BUCKET"}[request.param]
    if variable not in os.environ:
        pytest.skip(f"set {variable} to run against this service")
    root, options, endpoint = _s3_service() if request.param == "s3" else _hf_bucket()
    prefix = f"{root}/geosave-tests/{uuid.uuid4()}"

    # GDAL reads TIFFs itself, so it is told about the same service.
    before = {name: os.environ.get(name) for name in _GDAL_KEYS}
    address = urlsplit(endpoint)
    configure_gdal(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        aws_default_region="us-east-1",
        aws_s3_endpoint=address.netloc,
        aws_virtual_hosting=False,
        aws_https=address.scheme == "https",
        gdal_disable_readdir_on_open="EMPTY_DIR",
    )
    try:
        yield prefix, options
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        folder = storage.location(prefix, options)
        if folder.exists():
            folder.fs.rm(folder.path, recursive=True)
