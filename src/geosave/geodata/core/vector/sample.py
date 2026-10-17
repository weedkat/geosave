"""The `gs` accessor of one sample-table row: Items read lazily as a DataTree.

Every collection keeps its native grid and selected acquisition dates. Chips
crop and pad each collection on its native pixels. Alignment happens only
on request; frame construction owns temporal selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import hashlib
import numpy as np
import shapely
from odc.geo.geom import Geometry
import orjson
import pandas as pd
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

if TYPE_CHECKING:
    import pystac
    from shapely.geometry.base import BaseGeometry
    from odc.geo import SomeResolution

    from geosave.geodata import DataTree
    from geosave.geodata.core.anchor import GeoAnchor
    from geosave.geodata.io.raster.stac import ChunkSpec
    from geosave.geodata.transform.extent import Padding
    from geosave.geodata.transform.warp import Extent, Native, Resampling, Target


class GeoSample:
    """Read one row of a sample table as a tree, one group per collection.

    Args:
        row: One row of a sample table.

    Examples:
        >>> row = chips.iloc[0]
        >>> list(row.gs.items)
        ['label', 'dem', 's2']
        >>> row.gs.to_xarray().gs.groups
        ('label', 'dem', 's2')
    """

    def __init__(self, row: pd.Series) -> None:
        """Bind the accessor to one sample row."""
        self._row = row

    @classmethod
    def from_items(
        cls, items: Mapping[str, list[pystac.Item]], *, footprint: BaseGeometry
    ) -> pd.Series:
        """Build a gridless spatial sample; its Items retain their own dates."""
        from .query import read_item_spans

        starts, ends = [], []
        for entries in items.values():
            first, last = read_item_spans(entries)
            starts.extend(first.dropna().tolist())
            ends.extend(last.dropna().tolist())
        footprint = footprint.normalize()
        row = pd.Series(
            {
                "id": hashlib.sha256(footprint.wkb).hexdigest()[:20],
                "area": shapely.to_geojson(footprint),
                "geometry": footprint,
                "parent": None,
            }
        )
        span = (min(starts), max(ends)) if starts else None
        return cls(row)._to_frame(items, timespan=span)

    @property
    def key(self) -> str:
        """Return spatial identity, including chip grid and padding when present."""
        grid = self.geobox
        if grid is None:
            return str(self._row["id"])
        return orjson.dumps(
            [
                str(self._row["id"]),
                str(grid.crs),
                list(grid.shape),
                [float(value) for value in grid.transform[:6]],
                self.pad_widths,
            ]
        ).decode()

    def _to_frame(
        self,
        items: Mapping[str, list[pystac.Item]],
        *,
        timespan: tuple[pd.Timestamp, pd.Timestamp] | None,
    ) -> pd.Series:
        """Copy the row with selected Items and dates, preserving its other fields.

        Args:
            items: Selected Items grouped by collection, in date order.
            timespan: Inclusive frame bounds, or None for a timeless sample.

        Returns:
            Native Series with the selected Items and span.
        """
        records = {}
        for name, entries in items.items():
            records[name] = [
                entry.to_dict(include_self_link=False, transform_hrefs=False)
                for entry in entries
            ]

        row = self._row.copy()
        row["collections"] = list(records)
        row["items"] = orjson.dumps(records).decode()
        row["start_datetime"] = None if timespan is None else timespan[0]
        row["end_datetime"] = None if timespan is None else timespan[1]
        return row

    def _to_chip(
        self,
        grid: GeoBox,
        *,
        region: GeoBox,
        pad_widths: list[list[int]],
        pad_mode: Padding,
        geometry: str = "geometry",
    ) -> pd.Series:
        """Copy the row onto a chip grid, recording its padding and parent.

        Args:
            grid: Chip grid on the chosen pixel lattice.
            region: Full region covered by this chip layout, for merging.
            pad_widths: Pixels before and after each of the y and x axes.
            pad_mode: How pixels outside the read region are filled.
            geometry: Active geometry column to update with the chip footprint.

        Returns:
            Native Series describing the chip and retaining the row's other fields.
        """
        row = self._row.copy()
        row["parent"] = self.key
        row["parent_grid"] = orjson.dumps(
            [
                str(region.crs),
                list(region.shape),
                [float(value) for value in region.transform[:6]],
            ]
        ).decode()
        row["proj:code"] = str(grid.crs)
        row["proj:transform"] = [float(value) for value in grid.transform[:6]]
        row["proj:shape"] = list(grid.shape)
        row["pad_widths"] = pad_widths
        is_padded = any(before or after for before, after in pad_widths)
        row["pad_mode"] = pad_mode if is_padded else None
        row[geometry] = grid.extent.to_crs("EPSG:4326").geom
        return row

    @property
    def parent_grid(self) -> GeoBox | None:
        """Return the chip layout's full grid, None for an unchipped sample."""
        value = self._row.get("parent_grid")
        if not isinstance(value, str):
            return None
        crs, shape, transform = orjson.loads(value)
        return GeoBox(tuple(shape), Affine(*transform), crs)

    @property
    def anchor(self) -> GeoAnchor | None:
        """Return a chip anchor, None for a gridless sample or frame."""
        from geosave.geodata.core.anchor import GeoAnchor

        return None if self.geobox is None else GeoAnchor(self.geobox, self.timespan)

    @property
    def region(self) -> Geometry:
        """Return the ground read from files, independent of the display geometry."""
        grid = self.geobox
        if grid is None:
            return Geometry(orjson.loads(str(self._row["area"])), "EPSG:4326")
        (top, bottom), (left, right) = self.pad_widths
        height, width = grid.shape
        return grid[top : height - bottom, left : width - right].extent

    @property
    def items(self) -> dict[str, list[pystac.Item]]:
        """Return the Items each collection reads, in date order."""
        import pystac

        items_by_collection = {}
        for collection, item_dicts in orjson.loads(str(self._row["items"])).items():
            collection_items = []
            for item_dict in item_dicts:
                collection_items.append(pystac.Item.from_dict(item_dict))
            items_by_collection[collection] = collection_items
        return items_by_collection

    @property
    def geobox(self) -> GeoBox | None:
        """Return the recorded chip grid, None before chipping."""
        if not isinstance(self._row.get("proj:code"), str):
            return None
        height, width = (int(edge) for edge in self._row["proj:shape"])
        transform = Affine(*self._row["proj:transform"][:6])
        return GeoBox((height, width), transform, str(self._row["proj:code"]))

    @property
    def timespan(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return the row's span as naive UTC, None when timeless."""
        from .query import to_naive_utc

        span_instants = to_naive_utc(
            cast("pd.Series", self._row[["start_datetime", "end_datetime"]])
        )
        if span_instants.isna().any():
            return None
        return span_instants.iloc[0], span_instants.iloc[1]

    @property
    def pad_widths(self) -> list[tuple[int, int]]:
        """Return the padded pixels before and after `y`, then `x`."""
        pad_widths = []
        widths = (
            self._row["pad_widths"] if "pad_widths" in self._row else [[0, 0], [0, 0]]
        )
        for before, after in widths:
            pad_widths.append((int(before), int(after)))
        return pad_widths

    def to_xarray(
        self,
        *,
        collections: Sequence[str] | None = None,
        target: Target | None = None,
        extent: Extent = "match",
        resolution: SomeResolution | Native | None = None,
        resampling: Resampling | Mapping[str, Resampling] | None = None,
        chunks: ChunkSpec = {},
    ) -> DataTree:
        """Open the sample as a lazy xarray DataTree, keeping native grids.

        Each collection reads its selected Items on its own pixels. Chips crop
        to their read region and pad as recorded. Actual acquisition dates are
        preserved for unframed samples. Frames execute their recorded bin
        reductions and interpolation, returning one time step per requested bin.
        An unframed collection static over the span reads without a time
        dimension; other collections retain their time axis::

            row grid (chip)        s2, 10 m          s1, 20 m
            +--pad--+------+       32 x 32           16 x 16
            |       |region|       native            native
            +-------+------+       target=grid  ->   32 x 32 each

        Args:
            collections: Collections to read, in output order. None reads all.
            target: Collection name, GeoBox, reference raster, or CRS to align
                onto. A collection name uses that collection's cropped and
                padded grid, even when excluded from `collections`. Matching
                collection names take precedence over CRS strings. None keeps
                native grids.
            extent: Ground to cover when aligning: ``"match"`` adopts the
                target, ``"union"`` covers all groups, ``"intersection"``
                keeps their shared extent.
            resolution: Output pixel size, or ``"native"`` to keep each
                group's resolution. Requires a target.
            resampling: Kernel, or mapping of ``collection/variable`` to
                kernels with ``"*"`` as fallback. None uses nearest when
                aligning. Requires a target.
            chunks: File-opening chunk sizes. An empty mapping uses native
                Dask chunks; ``"auto"`` chooses sizes; None disables Dask.

        Returns:
            DataTree, even for one selected collection, with one Dataset per
            collection, each carrying its
            Collection's ACDD attrs where the table holds the definition.

        Raises:
            KeyError: A selected collection is not in the row.
            ValueError: Selection is empty or repeated, or alignment options
                are given without a target.
            TypeError: `collections` is a string rather than a sequence.

        Examples:
            >>> tree = row.gs.to_xarray(target="label")
            >>> tree["s1"].dataset.sizes
            Frozen({'time': 2, 'y': 32, 'x': 32})
        """
        from geosave.geodata.core.raster.create import tree

        if target is None and (
            resampling is not None or resolution is not None or extent != "match"
        ):
            raise ValueError("alignment options require an explicit target")

        items_by_collection = self.items
        if isinstance(collections, str):
            raise TypeError("collections takes a sequence, such as ['s2']")
        names = list(items_by_collection) if collections is None else list(collections)
        if not names or len(names) != len(set(names)):
            raise ValueError("select at least one collection, each once")
        if missing_names := set(names) - items_by_collection.keys():
            raise KeyError(f"unknown collections: {sorted(missing_names)}")

        rasters_by_collection: dict[str, xr.Dataset] = {}
        grid = (
            target
            if isinstance(target, GeoBox) and resolution is None and extent == "match"
            else None
        )
        recipe = self._row.get("frame")
        bins = orjson.loads(recipe) if isinstance(recipe, str) else {}
        for collection in names:
            if collection in bins:
                rasters_by_collection[collection] = self._read_frame(
                    collection,
                    items_by_collection[collection],
                    bins[collection],
                    chunks,
                    grid=grid,
                    resampling="nearest" if resampling is None else resampling,
                )
                continue
            rasters_by_collection[collection] = self._read_raster(
                collection,
                items_by_collection[collection],
                chunks,
                grid=grid,
                resampling="nearest" if resampling is None else resampling,
            )
        rasters = tree(rasters_by_collection)

        if target is None:
            return rasters
        if isinstance(target, str) and target in items_by_collection:
            if target in rasters_by_collection:
                target = rasters_by_collection[target]
            else:
                target = self._read_raster(target, items_by_collection[target], chunks)
        return rasters.gs.align(
            target=target,
            extent=extent,
            resolution=resolution,
            resampling="nearest" if resampling is None else resampling,
        )

    def _read_frame(
        self,
        collection: str,
        items: list[pystac.Item],
        recipe: dict,
        chunks: ChunkSpec,
        *,
        grid: GeoBox | None,
        resampling: Resampling | Mapping[str, Resampling],
    ) -> xr.Dataset:
        """Read each selected bin, reduce it, then fill interior absent bins."""
        from geosave.geodata.transform import nodata, temporal

        rasters = []
        present = []
        starts = []
        ends = []
        for start, end, positions in recipe["slots"]:
            starts.append(np.datetime64(start, "us"))
            ends.append(np.datetime64(end, "us") - np.timedelta64(1, "us"))
            if not positions:
                continue
            selected = [items[position] for position in positions]
            if recipe["reduce"] == "first":
                selected = selected[:1]
            elif recipe["reduce"] == "last":
                selected = selected[-1:]
            observations = []
            for item in selected:
                observation = self._read_raster(
                    collection, [item], chunks, grid=grid, resampling=resampling
                )
                if "time" not in observation.dims:
                    observation = observation.expand_dims("time")
                observations.append(observation)
            raster = temporal.concat(observations).sortby("time")
            if raster.sizes["time"] > 1 and recipe["reduce"] is None:
                raise ValueError(
                    f"{collection!r} bin has multiple raster time steps; specify reduce"
                )
            raster = temporal.reduce(raster, recipe["reduce"] or "first")
            raster = raster.drop_vars(
                [
                    name
                    for name in ("time", "start_datetime", "end_datetime")
                    if name in raster.coords
                ]
            )
            rasters.append(raster.expand_dims(time=[starts[-1]]))
            present.append(starts[-1])
        raster = temporal.concat(rasters).reindex(time=starts)
        raster = raster.assign_coords(
            start_datetime=("time", starts), end_datetime=("time", ends)
        )
        raster.time.attrs = {}
        if recipe["fill"] == "interpolate" and len(present) < len(starts):
            filled = temporal.interpolate(nodata.to_nan(raster), "linear")
            observed = xr.DataArray(
                np.isin(starts, present), dims="time", coords={"time": starts}
            )
            raster = raster.where(observed, filled)
        return raster

    def _read_raster(
        self,
        collection: str,
        items: list[pystac.Item],
        chunks: ChunkSpec,
        *,
        grid: GeoBox | None = None,
        resampling: Resampling | Mapping[str, Resampling] = "nearest",
    ) -> xr.Dataset:
        """Open selected Items, aligning before assembly when given a fixed grid."""
        from geosave.geodata.io.raster.stac import open_raster
        from geosave.geodata.transform import extent

        collection_fields = self._row.attrs.get("stac:collections", {})
        if isinstance(resampling, Mapping):
            kernels = {}
            for name, kernel in resampling.items():
                if name == "*":
                    kernels[name] = kernel
                elif name.startswith(f"{collection}/"):
                    kernels[name[len(collection) + 1 :]] = kernel
            resampling = kernels
        raster = open_raster(
            items,
            geobox=grid,
            resampling=resampling,
            chunks=chunks,
            collection_fields=collection_fields.get(collection),
        )
        # Frame construction owns Item selection. Re-filtering by the output
        # frame span would discard history and forecast targets.
        if not self._is_static(items, self.timespan) and "time" not in raster.dims:
            raster = raster.expand_dims("time")
        if self.geobox is not None:
            raster = extent.crop(raster, self.region, mask=False)

        return self._pad_region(raster)

    def _is_static(
        self,
        items: list[pystac.Item],
        timespan: tuple[pd.Timestamp, pd.Timestamp] | None,
    ) -> bool:
        """Tell whether every Item is timeless or covers the row's whole span."""
        from .query import read_item_spans

        start_instants, end_instants = read_item_spans(items)
        if bool(start_instants.isna().all()):
            return True
        if timespan is None:
            return False

        covers_span = (start_instants <= timespan[0]) & (end_instants >= timespan[1])
        return bool((start_instants.isna() | covers_span).all())

    def _pad_region(self, raster: xr.Dataset) -> xr.Dataset:
        """Pad to the row's ground extent on the raster's native pixel lattice."""
        from geosave.geodata.transform.extent import pad

        pad_mode = self._row.get("pad_mode")
        if not isinstance(pad_mode, str):
            return raster

        grid = raster.gs.anchor.geobox
        chip_grid = self.geobox
        if chip_grid is None:
            return raster
        padded_grid = grid.enclosing(chip_grid.extent)
        offset = ~grid.transform * padded_grid.transform
        top, left = max(0, -round(offset.f)), max(0, -round(offset.c))
        bottom = max(0, padded_grid.height - grid.height - top)
        right = max(0, padded_grid.width - grid.width - left)

        return pad(
            raster, [(top, bottom), (left, right)], mode=cast("Padding", pad_mode)
        )
