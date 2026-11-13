# ruff: noqa
# Throwaway prototype for docs/contributing/design/catalog-to-training.md; not library code.
"""Prototype: a catalog as a lazily indexed xarray cube on a declared grid.

No dask graph: each variable wraps a BackendArray. Indexing narrows the grid and
the time groups; `.values` performs one ODC read onto exactly that window.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import odc.stac
import pandas as pd
import pystac
import shapely
import shapely.geometry
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from xarray.backends import BackendArray
from xarray.core import indexing

import geosave as gs  # noqa: F401


@dataclass(frozen=True)
class Read:
    """One ODC read: which Items fill which output time position."""

    items: tuple[pystac.Item, ...]
    positions: tuple[int, ...]  # time group of each Item


class CollectionReader:
    """Reads every selected band of one collection for one window, once.

    Sibling variables index with the same key in turn (`ds.isel(...).load()`),
    so the last window's bands are kept and the next variable reuses them.
    """

    def __init__(self, groups, bands, grid, resampling, pool):
        self.groups = groups
        self.bands = bands
        self.grid = grid
        self.resampling = resampling
        self.pool = pool
        self.reads = 0
        self._recent = {}
        self._lock = threading.Lock()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_recent"] = {}
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def read(self, positions: tuple[int, ...], y0, y1, x0, x1) -> dict[str, np.ndarray]:
        request = (positions, y0, y1, x0, x1)
        with self._lock:
            if request in self._recent:
                return self._recent[request]
            window = self.grid[y0:y1, x0:x1]
            items = []
            group_of = {}
            for position in positions:
                for item in self.groups[position]:
                    items.append(item)
                    group_of[item.id] = position
            self.reads += 1
            loaded = odc.stac.load(
                items,
                bands=list(self.bands),
                geobox=window,
                groupby=lambda item, parsed, index: group_of[item.id],
                chunks=None,
                resampling=self.resampling,
                pool=self.pool,
            )
            # ODC orders groups by their first Item's time, as positions are.
            pixels = {}
            for band in self.bands:
                values = loaded[band].values
                if values.ndim == 2:
                    values = values[np.newaxis]
                pixels[band] = dict(zip(sorted(positions), values, strict=True))
            self._recent[request] = pixels
            while len(self._recent) > 16:
                self._recent.pop(next(iter(self._recent)))
            return pixels


class CatalogArray(BackendArray):
    """One band, (time, y, x) on the reader's grid."""

    def __init__(self, reader: CollectionReader, band, dtype, nodata):
        self.reader = reader
        self.band = band
        self.dtype = np.dtype(dtype)
        self.nodata = nodata
        self.shape = (len(reader.groups), reader.grid.shape.y, reader.grid.shape.x)

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._read
        )

    def _read(self, key):
        times, rows, cols = key
        positions = np.arange(self.shape[0])[times]
        ys = np.arange(self.shape[1])[rows]
        xs = np.arange(self.shape[2])[cols]
        squeeze = []
        for axis, chosen in enumerate((positions, ys, xs)):
            if np.ndim(chosen) == 0:
                squeeze.append(axis)
        positions = np.atleast_1d(positions)
        ys = np.atleast_1d(ys)
        xs = np.atleast_1d(xs)
        out = np.full((len(positions), len(ys), len(xs)), self.nodata, self.dtype)
        if len(positions) and len(ys) and len(xs):
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            unique = tuple(sorted({int(p) for p in positions}))
            stack = self.reader.read(unique, y0, y1, x0, x1)[self.band]
            for index, position in enumerate(positions):
                out[index] = stack[int(position)][np.ix_(ys - y0, xs - x0)]
        return out.squeeze(axis=tuple(squeeze)) if squeeze else out


def item_time(item: pystac.Item) -> pd.Timestamp:
    if item.datetime is not None:
        return pd.Timestamp(item.datetime).tz_convert(None)
    return pd.Timestamp(item.properties["start_datetime"]).tz_convert(None)


def solar_day(item: pystac.Item) -> pd.Timestamp:
    lon = (item.bbox[0] + item.bbox[2]) / 2
    return (item_time(item) + pd.Timedelta(hours=int(lon / 15))).normalize()


def open_cube(
    rows,
    grid: GeoBox,
    *,
    bands=None,
    resampling="nearest",
    pool=None,
) -> xr.Dataset:
    """Rows of one collection as a lazily indexed (time, y, x) Dataset on `grid`."""
    footprint = grid.extent.to_crs("EPSG:4326").geom
    rows = rows[rows.intersects(footprint)]
    items = rows.gs.to_items()
    by_day: dict[pd.Timestamp, list[pystac.Item]] = {}
    for item in sorted(items, key=item_time):
        by_day.setdefault(solar_day(item), []).append(item)
    groups = list(by_day.values())
    (parsed,) = odc.stac.parse_items([items[0]])
    names = parsed.collection.normalize_band_query(bands)
    times = []
    starts = []
    ends = []
    for group in groups:
        stamps = [item_time(item) for item in group]
        times.append(min(stamps))
        start = group[0].properties.get("start_datetime")
        end = group[0].properties.get("end_datetime")
        starts.append(pd.Timestamp(start).tz_convert(None) if start else min(stamps))
        ends.append(pd.Timestamp(end).tz_convert(None) if end else max(stamps))
    variables = {}
    reader = CollectionReader(groups, names, grid, resampling, pool)
    for name in names:
        meta = parsed.collection[name]
        nodata = meta.nodata if meta.nodata is not None else 0
        array = CatalogArray(reader, name, meta.data_type, nodata)
        data = indexing.LazilyIndexedArray(array)
        attrs = {"_FillValue": np.array(nodata, dtype=meta.data_type).item()}
        variables[name] = xr.Variable(("time", "y", "x"), data, attrs=attrs)
    coords = dict(xr_coords(grid))
    coords["time"] = ("time", pd.DatetimeIndex(times))
    coords["start_datetime"] = ("time", pd.DatetimeIndex(starts))
    coords["end_datetime"] = ("time", pd.DatetimeIndex(ends))
    ids = []
    for group in groups:
        ids.append(" ".join(item.id for item in group))
    coords["ids"] = ("time", np.array(ids, dtype=object))
    footprints = []
    for group in groups:
        union = shapely.union_all(
            [shapely.geometry.shape(item.geometry) for item in group]
        )
        footprints.append(shapely.to_wkb(union))
    coords["footprint"] = ("time", np.array(footprints, dtype=object))
    return xr.Dataset(variables, coords=coords)


def open_tree(catalog, grid: GeoBox, **options) -> xr.DataTree:
    """Every collection of `catalog` as a cube node on one grid."""
    nodes = {}
    for collection, rows in catalog.groupby("collection"):
        nodes[str(collection)] = open_cube(rows, grid, **options)
    return xr.DataTree.from_dict(nodes)


def select_slots(cube: xr.Dataset, slots: list[str]) -> xr.Dataset:
    """Index `cube` by Item ids, one time step per slot; "" marks a missing slot.

    Reused Items repeat; missing slots read position 0 and are masked to nodata,
    with a boolean `valid` coordinate along time.
    """
    position_of = {}
    for position, joined in enumerate(cube["ids"].values):
        for identity in joined.split(" "):
            position_of[identity] = position
    positions = []
    valid = []
    for identity in slots:
        positions.append(position_of.get(identity, 0))
        valid.append(identity in position_of)
    picked = cube.isel(time=positions).assign_coords(valid=("time", valid))
    return picked


def coverage(cube: xr.Dataset, chips) -> xr.DataArray:
    """Fraction of each chip inside each time step's footprint, from coordinates only."""
    from odc.geo.geom import Geometry

    crs = chips.crs
    shapes = np.asarray(chips.geometry.values)
    areas = shapely.area(shapes)
    rows = []
    for wkb in cube["footprint"].values:
        footprint = Geometry(shapely.from_wkb(wkb), crs="EPSG:4326").to_crs(crs).geom
        rows.append(shapely.area(shapely.intersection(shapes, footprint)) / areas)
    return xr.DataArray(
        np.array(rows).T, dims=("chip", "time"), coords={"time": cube.time}
    )
