from __future__ import annotations

from tests.conftest import combine_item, write_cogs

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
from geosave.geodata.stac.catalog import create_catalog
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
    from contextlib import ExitStack
    from datetime import datetime as DateTime

    import pystac
    from geosave.geodata import open_raster
    from geosave.geodata.stac.publish import create_stac_item

    with ExitStack() as opened:
        groups = {}
        paths = files.items() if isinstance(files, Mapping) else enumerate(flat(files))
        for key, path in paths:
            raster = opened.enter_context(open_raster(path))
            span = raster.gs.timespan or (DateTime(2025, 6, 1), DateTime(2025, 6, 1))
            group = None if isinstance(files, Mapping) else span
            if not isinstance(files, Mapping):
                key = (
                    str(next(iter(raster.data_vars)))
                    if len(raster.data_vars) == 1
                    else "image"
                )
            groups.setdefault(group, {})[str(key)] = (path, raster, span)

        for assets in groups.values():
            paths = {}
            spans = []
            for key, (path, raster, span) in assets.items():
                paths[key] = path
                spans.append(span)
            start = min(span[0] for span in spans)
            end = max(span[1] for span in spans)
            if datetime is not None:
                if isinstance(datetime, tuple):
                    start, end = datetime
                else:
                    start = end = datetime
            raster = next(iter(assets.values()))[1]
            values = {"start": start, "end": end}
            centroid = raster.gs.geobox.extent.to_crs("EPSG:4326").centroid.coords[0]
            values.update(lon=centroid[0], lat=centroid[1])
            properties = {}
            instant = start if start == end else None
            if instant is None:
                properties = {
                    "start_datetime": pd.Timestamp(start).isoformat(),
                    "end_datetime": pd.Timestamp(end).isoformat(),
                }
            entry = create_stac_item(
                paths,
                id=id_format.format_map(values),
                input_datetime=instant,
                properties=properties,
                collection=collection.id
                if isinstance(collection, pystac.Collection)
                else collection,
                with_proj=True,
                with_raster=True,
                with_eo=True,
            )
            if isinstance(collection, pystac.Collection):
                entry.set_collection(collection)
            yield entry


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
    from pathlib import PurePath

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
            item.id = name if prefix is None else f"{prefix}-{name}"
            items.append(item)
    return create_catalog(
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
    return raw


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
        raster = sample[name].gs.combine_stac()
        templates[name] = cog_template(name, dated="time" in raster.dims)
    return templates


def write_collection_cogs(tree, root, templates=None) -> dict[str, list]:
    """Write each collection of a tree as COGs, keyed by collection, as `item_table` takes them.

    A template naming `{variable}` writes a file per variable; otherwise every
    variable of a date shares one file, as one asset.
    """
    templates = templates or tree_templates(tree)
    files = {}
    for name in tree.gs.groups:
        raster = tree[name].gs.combine_stac()
        template = templates[name]
        files[name] = write_cogs(raster, root, template)
    return files


def read_items(rows):
    """Read one collection's rows as one raster: each a scene, joined along time."""
    from geosave.geodata.stac.load import load_datatree
    from geosave.geodata import attrs

    scenes = []
    for entry in rows.gs.to_items():
        scenes.append(combine_item(load_datatree([entry])))
    if len(scenes) == 1:
        return scenes[0]
    return xr.concat(
        scenes, dim="time", join="exact", combine_attrs=attrs.combine_attrs
    )


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


@pytest.fixture
def nested_tree():
    """Nested measurements with parent-node data and repeated grid labels."""
    source = (
        build_raster(times=2).chunk().assign_coords(cloud_fraction=("time", [0.1, 0.2]))
    )
    return xr.DataTree.from_dict(
        {
            "s2": xr.Dataset(attrs={"source": "sensor"}),
            "s2/measurements": xr.Dataset(attrs={"description": "organization"}),
            "s2/measurements/native": source,
            "s2/measurements/native/qa": source[["red"]].rename(red="cloud"),
            "s2/quality/native": source[["red"]].rename(red="confidence"),
            "s2/empty": xr.Dataset(attrs={"description": "no raster"}),
            "dem/native": build_raster()[["red"]].rename(red="height").chunk(),
        }
    )


def assign_cells(raster, pairs):
    """Assign one stated start/end pair per time label, as a reader would."""
    labels = raster.gs.time_coordinate
    starts = np.array([start for start, _ in pairs], dtype="datetime64[ns]")
    ends = np.array([end for _, end in pairs], dtype="datetime64[ns]")
    return raster.assign_coords(
        start_datetime=(labels.dims, starts.reshape(labels.shape)),
        end_datetime=(labels.dims, ends.reshape(labels.shape)),
    )


def build_dated_raster() -> xr.Dataset:
    """Build a lazy two-date raster whose cells start before its first label.

    Labels carry nanoseconds, so a round trip that loses precision shows.
    """
    raster = build_raster(times=2).chunk(time=1)
    raster = raster.assign_coords(
        time=raster.time.copy(
            data=np.array(
                ["2025-01-10T10:30:00.123456789", "2025-01-20T10:30:00"],
                "datetime64[ns]",
            )
        )
    )
    return assign_cells(
        raster,
        [("2025-01-01", "2025-01-15"), ("2025-01-15", "2025-02-01")],
    ).chunk(time=1)
