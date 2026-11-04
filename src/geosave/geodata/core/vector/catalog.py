"""Explore, query and sample spatial catalog records on native GeoDataFrames."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from os import PathLike
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave.geodata.utils.datetime import (
    AnchorDatetime,
    DateRange,
    find_overlapping,
    parse_daterange,
    read_row_spans,
)
from geosave.geodata.utils.geometry import SomeGeometry, to_shapely

from geosave.geodata.stac.catalog import FOOTER_KEY
from geosave.geodata.stac.catalog import ITEM_COLUMNS


if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox
    import holoviews as hv
    import pystac
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack
    from upath import UPath
    from geosave.geodata import DataTree
    from geosave.geodata.io.storage import Location
    from geosave.geodata.io.formats.geojson import GeoJSONWriteOptions
    from geosave.geodata.io.formats.geopackage import GeoPackageWriteOptions
    from geosave.geodata.io.formats.geoparquet import GeoParquetWriteOptions

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core.anchor import GeoAnchor
    from geosave.geodata.viz.explore import Explorer

    from .samples import GeoSamples

type SpatialPredicate = Literal[
    "intersects", "within", "contains", "covers", "covered_by"
]

SPATIAL_PREDICATES = get_args(SpatialPredicate.__value__)


class GeoCatalog:
    """Place, draw, query, write and sample spatial records, including STAC Items.

    The accessor of every GeoDataFrame, which it leaves as it is: geometry with
    information attached, an active geometry column, a CRS, and any other
    columns. Methods fall in two tiers.

    - Spatial records need only a geometry: `crs`, `footprint`, `query`,
      `sample`, `plot`, `explore`, `rasterize`, `concat`. Querying by date uses
      the optional `datetime`, `start_datetime` and `end_datetime` columns. A
      GeoJSON, GeoPackage or GeoParquet file of plain features is this.
    - Item tables add the stac-geoparquet columns, one Item per row, with the
      Collections in the `stac-geoparquet` attrs travelling with the rows:
      `to_items`, `to_datatree`, `collections`, and a row's `anchor`. They come
      from `to_catalog`, `StacClient.search` or a stac-geoparquet file, and
      these methods check their columns when called.

    Raises:
        AttributeError: `data` is not a GeoDataFrame, or has no active
            geometry column.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Bind the accessor to one GeoDataFrame.

        Args:
            data: GeoDataFrame with an active geometry column.

        Raises:
            AttributeError: The input is not a GeoDataFrame or has no active
                geometry column.
        """
        # pandas reads AttributeError as "this object has no such accessor".
        if not isinstance(data, gpd.GeoDataFrame):
            raise AttributeError(
                f"gs reads a GeoDataFrame, got {type(data).__name__}; build one "
                f"with gpd.GeoDataFrame(...)"
            )
        if data.active_geometry_name is None:
            raise AttributeError("gs needs an active geometry column")
        self._data = data

    @property
    def crs(self) -> OdcCRS:
        """Return the coordinate reference system every geometry shares."""
        if self._data.crs is None:
            raise ValueError("this vector has no CRS")
        return OdcCRS(self._data.crs)

    @property
    def footprint(self) -> Geometry:
        """Return the union of all geometries in the vector CRS.

        Returns:
            ODC geometry carrying the vector CRS.

        Raises:
            ValueError: The frame holds no row.
        """
        if self._data.empty:
            raise ValueError("an empty vector has no footprint")
        return Geometry(self._data.geometry.union_all(), crs=self.crs)

    @property
    def anchor(self) -> GeoAnchor:
        """Return feature union and the temporal envelope of dated rows.

        Timeless rows do not widen the envelope. No assets are opened and no
        pixel grid is inferred. Invalid geometry or incomplete dates raise.
        """
        from geosave.geodata.core.anchor import GeoAnchor
        from geosave.geodata.stac.item import read_timespan

        anchor = GeoAnchor.from_geometry(self._data)
        columns = [
            key
            for key in ("datetime", "start_datetime", "end_datetime")
            if key in self._data
        ]
        spans = []
        for values in self._data[columns].itertuples(index=False, name=None):
            properties = {}
            for key, value in zip(columns, values, strict=True):
                if not pd.isna(value):
                    properties[key] = value
            start = properties.get("start_datetime", properties.get("datetime"))
            end = properties.get("end_datetime", properties.get("datetime"))
            if (start is None) != (end is None):
                raise ValueError("anchor rows need complete, ordered temporal bounds")
            span = read_timespan({"properties": properties})
            if span is not None:
                if span[0] > span[1]:
                    raise ValueError(
                        "anchor rows need complete, ordered temporal bounds"
                    )
                spans.append(span)
        if spans:
            return anchor.replace(
                timespan=(
                    min(start for start, _ in spans),
                    max(end for _, end in spans),
                )
            )
        return anchor

    @classmethod
    def concat(cls, vectors: Iterable[gpd.GeoDataFrame]) -> GeoDataFrame:
        """Concatenate spatial records without interpreting their identity.

        Args:
            vectors: Frames already expressed on one CRS plane.

        Returns:
            New frame with unioned property columns, a fresh RangeIndex, and
            its geometry column named `geometry`.

        Raises:
            ValueError: No frames are supplied, their CRSs differ, or a frame
                holds a second column named `geometry` beside its active one.
        """
        items = list(vectors)
        if not items:
            raise ValueError("concat needs at least one GeoDataFrame")
        crs = OdcCRS(items[0].crs)
        mismatches = [str(item.crs) for item in items if OdcCRS(item.crs) != crs]
        if mismatches:
            raise ValueError(
                f"the vectors' CRSs differ from {crs}: {mismatches}; "
                "call to_crs explicitly"
            )
        frames: list[gpd.GeoDataFrame] = []
        for item in items:
            frame = item.copy()
            if frame.active_geometry_name != "geometry":
                if "geometry" in frame.columns:
                    raise ValueError(
                        "cannot concatenate an active geometry with another "
                        "column named 'geometry'"
                    )
                frame = frame.rename_geometry("geometry")
            frames.append(frame)
        result = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs
        )
        collections = {}
        for frame in items:
            for key, value in (
                frame.attrs.get(FOOTER_KEY, {}).get("collections", {}).items()
            ):
                collections.setdefault(key, value)
        if any(FOOTER_KEY in frame.attrs for frame in items):
            result.attrs[FOOTER_KEY] = {"collections": deepcopy(collections)}
        return cast("GeoDataFrame", result)

    def plot(self, *, color: str = "tab:blue", alpha: float = 0.3) -> hv.Overlay:
        """Draw this vector's geometry in its own coordinates.

        Polygons draw outlined and filled, lines and points in `color`. The
        drawing overlays a raster plot with `*` where both share a CRS.

        Args:
            color: Colour of outlines, lines and points.
            alpha: Opacity of the polygon fill.

        Returns:
            Overlay of one element per geometry type the vector holds.

        Raises:
            ValueError: The vector holds no geometry or a GeometryCollection.

        Examples:
            >>> plots.gs.plot()
            >>> scene.gs.plot() * plots.to_crs(anchor.crs).gs.plot()
        """
        from geosave.geodata.viz.plot import draw_vector

        return draw_vector(self._data, color=color, alpha=alpha)

    def explore(
        self,
        *,
        column: str | None = None,
        basemap: str | None = None,
        color: str = "tab:blue",
        alpha: float = 0.3,
    ) -> Explorer:
        """Open this vector on a web map with a sidebar to explore it.

        Features have hover and click identification. Property and date filters
        narrow the table and map. Checking catalog rows opens their rasters;
        unchecking removes them.

        Args:
            column: Column that colours the features, or None for one colour.
            basemap: Basemap under the layers, as `viz.explore.page.map.BASEMAPS` names it.
                None uses the light basemap.
            color: Colour of every feature when no column colours them.
            alpha: Opacity of polygon fill; 0 draws boundaries only.

        Returns:
            Explorer, shown by a notebook or served with `.servable()`.

        Raises:
            KeyError: `column` is not a column of the frame.
            ValueError: The vector has no CRS, holds no geometry, or holds a
                GeometryCollection.

        Examples:
            >>> fields.gs.explore(column="crop")
            >>> fields.gs.explore(alpha=0.0, basemap="Imagery")  # boundaries only
        """
        from geosave.geodata.viz.explore import Explorer, create_layer

        layer = create_layer(self._data, column=column, color=color, alpha=alpha)
        return Explorer(layer, basemap=basemap)

    def rasterize(
        self,
        like: GeoBox | xr.DataArray | xr.Dataset | xr.DataTree,
        *,
        column: str | None = None,
        fill: int | float | bool = 0,
        dtype: DTypeLike | None = None,
        all_touched: bool = False,
    ) -> xr.DataArray:
        """Burn geometries or one property onto an existing exact grid.

        Args:
            like: Anchor or geolocated xarray object supplying the target grid.
            column: Property to burn. None returns a boolean presence mask.
            fill: Value outside geometries. A presence mask takes only 0 or
                False, since it is False outside them.
            dtype: Output dtype for property values.
            all_touched: Burn every touched pixel instead of pixel centres.

        Returns:
            Eager geolocated array on the exact target grid. A dated anchor
            dates it with a scalar `time` at the start of its timespan.

        Raises:
            KeyError: The selected property is absent.
            ValueError: A geometry is null, empty, or invalid, values or fill
                cannot be represented by the dtype, or `column` is None with a
                truthy `fill`.
        """
        from geosave.geodata.transform.vector import rasterize

        return rasterize(
            self._data,
            like,
            column=column,
            fill=fill,
            dtype=dtype,
            all_touched=all_touched,
        )

    @property
    def collections(self) -> dict[str, pystac.Collection]:
        """Return the Collections the table's footer metadata holds, by id."""
        import pystac

        footer = self._data.attrs.get(FOOTER_KEY, {})
        collections = {}
        for collection_id, fields in footer.get("collections", {}).items():
            collections[collection_id] = pystac.Collection.from_dict(fields)
        return collections

    def query(
        self,
        where: GeoAnchor
        | xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | SomeGeometry
        | None = None,
        *,
        when: AnchorDatetime | None = None,
        predicate: SpatialPredicate = "intersects",
    ) -> GeoDataFrame:
        """Select rows related to a place and a time span.

        One rule serves every vector, Items and features alike: the predicate
        reads from each row to `where`, and a row's span, from
        `start_datetime` and `end_datetime` each falling back to `datetime`,
        must share an instant with `when`. A missing bound is open, so an
        undated row matches any time.

        Args:
            where: Anchor or geolocated xarray object, whose ground and span
                are used; or a geometry: an odc Geometry carries its CRS, any
                other geometry is read as WGS84. None selects by time alone.
            when: Time span, as `parse_daterange` reads it, such as `"2024-03"`.
                None takes the span of an anchor `where`, else selects by
                place alone.
            predicate: Exact relation from each row to `where`.

        Returns:
            Matching original rows, preserving their index, order and metadata.

        Raises:
            ValueError: The predicate is unsupported, `where` carries no
                locatable grid, or a date column holds invalid values.

        Examples:
            >>> items.gs.query(scene)["id"].tolist()
            ['s2_20240308', 'label']
            >>> items.gs.query(field, when="2024-03")["id"].tolist()
            ['s2_20240308', 's2_20240328', 'label']
        """
        if predicate not in SPATIAL_PREDICATES:
            raise ValueError(
                f"predicate must be one of {list(SPATIAL_PREDICATES)}, got {predicate!r}"
            )
        ground, timespan = self._read_ground_and_timespan(where)
        if when is not None:
            timespan = parse_daterange(when)

        rows = self._data
        if ground is not None and not rows.empty:
            rows = self._select_by_ground(rows, ground, predicate)
        if timespan is not None and not rows.empty:
            start_instants, end_instants = read_row_spans(rows)
            rows = rows.loc[find_overlapping(start_instants, end_instants, timespan)]
        return cast("GeoDataFrame", rows.copy())

    @staticmethod
    def _read_ground_and_timespan(
        where: GeoAnchor
        | xr.DataArray
        | xr.Dataset
        | xr.DataTree
        | SomeGeometry
        | None,
    ) -> tuple[Geometry | None, DateRange | None]:
        """Return the ground a query target covers and the span it is dated by."""
        from geosave.geodata.core.anchor import GeoAnchor

        if isinstance(where, xr.DataArray | xr.Dataset | xr.DataTree):
            where = cast("GeoAnchor", where.gs.anchor)
        if isinstance(where, GeoAnchor):
            return where.geometry, where.timespan
        if where is None or isinstance(where, Geometry):
            return where, None
        return Geometry(to_shapely(where), crs="EPSG:4326"), None

    def _select_by_ground(
        self, rows: gpd.GeoDataFrame, ground: Geometry, predicate: SpatialPredicate
    ) -> gpd.GeoDataFrame:
        """Keep the rows whose geometry relates to `ground` as `predicate` says."""
        shape = ground.to_crs(self.crs).geom
        # The spatial index finds candidates by bounding box; the predicate decides.
        candidate_positions = sorted(set(rows.sindex.query(shape)))
        candidate_rows = rows.iloc[candidate_positions]
        matches = getattr(candidate_rows.geometry, predicate)(shape)
        return cast("gpd.GeoDataFrame", candidate_rows.loc[matches])

    def to_geojson(
        self,
        path: str | PathLike[str],
        *,
        overwrite: bool = False,
        **options: Unpack[GeoJSONWriteOptions],
    ) -> UPath:
        """Write this vector as one GeoJSON file.

        GeoJSON states coordinates in WGS84, so a projected vector is
        reprojected on the way out. Reach for `to_geopackage` or
        `to_geoparquet` to keep the vector's own CRS.

        Args:
            path: Output path or URL ending in `.geojson` or `.json`.
            overwrite: Replace an existing file when true.
            **options: GeoJSON write options.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath carrying
            its storage options.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geojson("plots.geojson")
            PosixUPath('plots.geojson')
        """
        self._require_features()
        from geosave.geodata.io.formats import geojson

        return geojson.write_geojson(self._data, path, overwrite=overwrite, **options)

    def to_geopackage(
        self,
        path: str | PathLike[str],
        *,
        layer: str | None = None,
        overwrite: bool = False,
        **options: Unpack[GeoPackageWriteOptions],
    ) -> UPath:
        """Write this vector as one layer in a GeoPackage.

        Args:
            path: Output path or URL ending in `.gpkg`.
            layer: Layer name. None names the layer after the file stem.
            overwrite: Replace an existing file when true.
            **options: GeoPackage write options.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath carrying
            its storage options.

        Raises:
            FileExistsError: The path exists and `overwrite` is false.
            ValueError: The suffix is wrong.

        Examples:
            >>> plots.gs.to_geopackage("survey.gpkg", layer="plots")
            PosixUPath('survey.gpkg')
        """
        self._require_features()
        from geosave.geodata.io.formats import geopackage

        return geopackage.write_geopackage(
            self._data, path, layer=layer, overwrite=overwrite, **options
        )

    def to_geoparquet(
        self,
        path: Location,
        *,
        collections: Iterable[pystac.Collection] | None = None,
        overwrite: bool = False,
        **options: Unpack[GeoParquetWriteOptions],
    ) -> UPath:
        """Write features or a STAC catalog to one GeoParquet file.

        Frames carrying `attrs["stac-geoparquet"]` use STAC encoding,
        preserving Collection definitions and portable asset hrefs. Ordinary
        features use GeoPandas encoding. Catalog metadata is checked at write
        time, so missing fields raise instead of losing their STAC encoding.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            collections: Override catalog Collection definitions by id. None
                preserves the frame's definitions; feature tables take none.
            overwrite: Replace an existing file when true.
            **options: GeoParquet/storage options. STAC catalogs require WKB
                geometry, a bbox and no serialized pandas index.

        Returns:
            UPath of the file, carrying its filesystem options.

        Raises:
            FileExistsError: The path exists and overwrite is false.
            ValueError: Catalog metadata or fields are invalid, the rows are
                samples, or options conflict with the STAC encoding.
            TypeError: An option is unsupported or repeated.

        Examples:
            >>> catalog.gs.to_geoparquet("data/catalog.parquet")
            PosixUPath('data/catalog.parquet')
            >>> plots.gs.to_geoparquet("plots.parquet", compression="zstd")
            PosixUPath('plots.parquet')
        """
        from geosave.geodata.io.formats import geoparquet

        return geoparquet.write_geoparquet(
            self._data, path, collections=collections, overwrite=overwrite, **options
        )

    def sample(
        self,
        *,
        pair: Sequence[str] | None = None,
        tolerance: str | pd.Timedelta = "0D",
        window: int = 1,
        step: int = 1,
        freq: str | None = None,
        keep: Literal["first", "last"] = "last",
        overlap: float = 0.9,
    ) -> GeoSamples:
        """Group Items into samples: by footprint, then into windows of paired Items.

        No file is opened. A sample is a group of this table's rows; samples
        may share rows, so a swath or a DEM belongs to several.

        Footprint. Visited from smallest to largest, a footprint covering at
        least `overlap` of a smaller one repeats it; the others are distinct.
        Each distinct footprint gathers every Item sharing at least `overlap`
        of its area, so a large footprint joins every one it covers::

            s2 tile A, every date   ▭▭▭▭           -> sample of tile A
            s2 tile B               ▭▭▭▭           -> sample of tile B
            s1 swath           ▭▭▭▭▭▭▭▭▭▭▭▭        -> joins A and B

        Pair. Each Item of the first `pair` collection is one pair, joined by
        the nearest Item of every other `pair` collection within `tolerance`::

            s2      ● Jan 5        ● Feb 4        ● Mar 8
            s1       ● Jan 6      ● Feb 3          ● Mar 9      tolerance="2D"
            pair    [Jan 5]       [Feb 4]        [Mar 8]

        Window. `window` consecutive slots form one sample, a new one every
        `step` slots, as `pandas.Series.rolling` takes rows. Without `freq` each
        pair is a slot; with it, a slot is a `resample` bucket, filled by its
        `keep` pair, and a window holding an empty bucket is skipped::

            freq="MS", window=2      [── Jan ──)[── Feb ──)[── Mar ──)
                                       Jan 5      Feb 4      Mar 8
            windows                  └──── 1 ────┘
                                                └──── 2 ────┘

        Timeless Items, such as a DEM, join every window of their footprint.
        Dated Items of a collection outside `pair` join none.

        Args:
            pair: Collections paired into one time step; the first one's Items
                set the pairs. None makes each footprint one sample.
            tolerance: Largest gap between a pair's Items, as `pd.Timedelta`
                reads it.
            window: Slots per sample.
            step: Slots between window starts.
            freq: Slot span as `resample` reads it, such as ``"MS"``. None
                makes each pair a slot.
            keep: Which pair fills a slot holding several.
            overlap: Fraction of a footprint's area an Item must share to join
                it, and a footprint must cover to repeat a smaller one.

        Returns:
            Samples of this table, each its rows: a window's pairs in time
            order, then its footprint's timeless Items.

        Raises:
            KeyError: A `pair` collection names no row's collection.
            ValueError: A window option is given without `pair`, a number is
                out of range, an Item's footprint has no area, or no footprint
                holds a complete window.

        Warns:
            DroppedRowsWarning: Items without a partner, windows holding an
                empty slot, and footprints holding fewer than `window` slots.

        Examples:
            >>> samples = items.gs.sample(pair=["s2", "s1"], tolerance="2D", window=3, freq="MS")
            >>> samples[0]["collection"].tolist()
            ['s2', 's1', 's2', 's1', 's2', 's1', 'dem']
            >>> samples[0].gs.to_datatree().gs.groups
            ('s2', 's1', 'dem')
            >>> explorer.add(samples.footprints, samples[0])
        """
        import shapely

        from .samples import GeoSamples

        if not 0 < overlap <= 1:
            raise ValueError(
                f"overlap is a fraction of a footprint's area, got {overlap}"
            )
        if pair is None and (window != 1 or step != 1 or freq is not None):
            raise ValueError(
                "window, step and freq cut windows of pairs; name them with pair="
            )
        if window < 1 or step < 1:
            raise ValueError(
                f"window and step count slots, got window={window}, step={step}"
            )
        if pair is not None:
            missing_fields = {"collection", "datetime"} - set(self._data.columns)
            if missing_fields:
                raise ValueError(f"pairing needs columns {sorted(missing_fields)}")
            missing_collections = set(pair) - set(self._data["collection"].dropna())
            if missing_collections:
                raise KeyError(f"no Item belongs to {sorted(missing_collections)}")

        # Index labels are row positions from here on, as GeoSamples selects them.
        items = cast("gpd.GeoDataFrame", self._data.reset_index(drop=True))
        # Items join a footprint by shared area, which a point or line has none of.
        has_no_area = shapely.area(items.geometry.to_numpy()) == 0
        if has_no_area.any():
            arealess_ids = items.loc[has_no_area, "id"].tolist()
            raise ValueError(
                f"sampling groups Items by shared area; {arealess_ids} have none"
            )

        footprints = self._find_distinct_footprints(items.geometry, overlap)
        members = self._find_footprint_items(items, footprints, overlap)
        if pair is None:
            indices = members.groupby("footprint")["row"].agg(list).tolist()
        else:
            largest_gap = cast("pd.Timedelta", pd.Timedelta(tolerance))
            indices = self._cut_windows(
                items, members, pair, largest_gap, window, step, freq, keep
            )
        return GeoSamples(self._data, indices)

    @staticmethod
    def _find_distinct_footprints(
        footprints: gpd.GeoSeries, overlap: float
    ) -> gpd.GeoSeries:
        """Return the footprints that do not cover `overlap` of a smaller one, in row order.

        Footprints are visited from smallest to largest, so a swath covering
        whole tiles never stands for a sample of its own: each tile does.
        """
        import numpy as np
        import shapely

        geometries = footprints.to_numpy()
        distinct_rows = []
        distinct_geometries = []
        for position in np.argsort(shapely.area(geometries), kind="stable"):
            geometry = geometries[position]
            # A footprint repeats a distinct one when it covers `overlap` of its area
            shared_areas = shapely.area(
                shapely.intersection(geometry, distinct_geometries)
            )
            minimum_areas = overlap * shapely.area(distinct_geometries)
            if not np.any(shared_areas >= minimum_areas):
                distinct_rows.append(footprints.index[position])
                distinct_geometries.append(geometry)
        return cast("gpd.GeoSeries", footprints.loc[sorted(distinct_rows)])

    @staticmethod
    def _find_footprint_items(
        items: gpd.GeoDataFrame, footprints: gpd.GeoSeries, overlap: float
    ) -> pd.DataFrame:
        """Return one row per footprint and Item sharing `overlap` of its area.

        Returns:
            `footprint`, the footprint's number in `footprints`, and `row`, the
            Item's row, sorted by both.
        """
        import shapely

        # 1. Find the Items intersecting each footprint
        footprint_frame = gpd.GeoDataFrame(
            {"footprint": range(len(footprints))},
            geometry=footprints.to_numpy(),
            crs=items.crs,
        )
        candidates = items[[items.geometry.name]].sjoin(
            footprint_frame, predicate="intersects"
        )

        # 2. Keep the candidates sharing at least `overlap` of the footprint's area
        footprint_numbers = candidates["footprint"].to_numpy()
        footprint_geometries = footprints.to_numpy()[footprint_numbers]
        shared_areas = shapely.area(
            shapely.intersection(candidates.geometry.to_numpy(), footprint_geometries)
        )
        joins = shared_areas >= overlap * shapely.area(footprint_geometries)
        members = pd.DataFrame(
            {
                "footprint": footprint_numbers[joins],
                "row": candidates.index.to_numpy()[joins],
            }
        )
        return members.sort_values(["footprint", "row"], ignore_index=True)

    @staticmethod
    def _cut_windows(
        items: gpd.GeoDataFrame,
        members: pd.DataFrame,
        pair: Sequence[str],
        tolerance: pd.Timedelta,
        window: int,
        step: int,
        freq: str | None,
        keep: Literal["first", "last"],
    ) -> list[list[int]]:
        """Cut each footprint's Items into windows of paired Items, as `sample` explains.

        Returns:
            For each window, its rows: its pairs in time order, then the
            footprint's timeless Items.
        """
        import warnings

        from geosave.geodata.warnings import DroppedRowsWarning

        # 1. Date each footprint's Items; pyarrow datetimes have no merge_asof
        #    or resample support
        times = pd.to_datetime(items["datetime"]).iloc[members["row"]]
        collection_names = items["collection"].iloc[members["row"]]
        members = members.assign(
            time=times.reset_index(drop=True),
            collection=collection_names.reset_index(drop=True),
        )
        is_dated = members["time"].notna()

        # 2. Pair each Item of the first collection with the nearest Item of
        #    every other, within its footprint
        is_first = is_dated & (members["collection"] == pair[0])
        pairs = members.loc[is_first, ["footprint", "time", "row"]]
        pairs = pairs.rename(columns={"row": pair[0]}).sort_values("time")
        for collection in pair[1:]:
            is_partner = is_dated & (members["collection"] == collection)
            partners = members.loc[is_partner, ["footprint", "time", "row"]]
            pairs = pd.merge_asof(
                pairs,
                partners.rename(columns={"row": collection}).sort_values("time"),
                on="time",
                by="footprint",
                direction="nearest",
                tolerance=tolerance,
            )
        complete_pairs = pairs.dropna().sort_values(["footprint", "time"])
        unpaired_count = len(pairs) - len(complete_pairs)

        # 3. Lay the pairs into slots; with `freq`, each bucket takes its `keep`
        #    pair, and an empty bucket holds NaN
        slots = complete_pairs
        if freq is not None:
            pairs_by_footprint = complete_pairs.set_index("time").groupby("footprint")
            buckets = pairs_by_footprint[list(pair)].resample(freq)
            bucket_pairs = buckets.last() if keep == "last" else buckets.first()
            slots = bucket_pairs.reset_index()

        # 4. Roll windows of `window` consecutive slots, a new one every `step`,
        #    each joined by its footprint's timeless Items
        indices = []
        broken_count = 0
        short_count = 0
        for footprint in members["footprint"].unique():
            slot_rows = slots.loc[slots["footprint"] == footprint, list(pair)]
            is_timeless = (members["footprint"] == footprint) & ~is_dated
            timeless_rows = members.loc[is_timeless, "row"].tolist()
            if len(slot_rows) < window:
                short_count += 1
            for start in range(0, len(slot_rows) - window + 1, step):
                window_rows = slot_rows.iloc[start : start + window]
                if window_rows.isna().any(axis=None):
                    broken_count += 1
                    continue
                # Row-major order: each pair's collections, pair after pair
                rows = window_rows.astype("int64").to_numpy().ravel().tolist()
                indices.append(rows + timeless_rows)

        if unpaired_count:
            warnings.warn(
                f"{unpaired_count} {pair[0]} Items have no partner within {tolerance}",
                DroppedRowsWarning,
                stacklevel=3,
            )
        if broken_count:
            warnings.warn(
                f"{broken_count} windows hold an empty slot",
                DroppedRowsWarning,
                stacklevel=3,
            )
        if short_count:
            warnings.warn(
                f"{short_count} footprints hold fewer than {window} slots",
                DroppedRowsWarning,
                stacklevel=3,
            )
        if not indices:
            raise ValueError(
                f"no footprint holds {window} consecutive pairs of {list(pair)}"
            )
        return indices

    def to_datatree(self, **options: Any) -> DataTree:
        """Open the Items lazily as `/{collection}/{res}`, each grid node on its native grid.

        Each row opens as one scene, as `GeoCatalogItem.to_datatree` opens it;
        a collection's scenes join along `time` in date order, node by node,
        and a collection of one timeless Item has no `time` axis. Collections
        follow their first appearance.

        Args:
            **options: Forwarded to each asset's GDAL reader, such as
                `chunks` or `mask_and_scale`.

        Returns:
            DataTree of collection nodes, each carrying its Collection and
            extension schemas as `StacMetadata` attrs, over their grid nodes;
            closing it closes every file.

        Raises:
            ValueError: An Item names no collection, or a collection's scenes
                differ in grid nodes, grids or variables.

        Examples:
            >>> tree = items.gs.query(when="2024-03-08").gs.to_datatree()
            >>> tree.gs.groups
            ('s2', 'dem')
            >>> list(tree.gs.rasters)
            ['s2/r10m', 's2/r20m', 'dem/r30m']
        """
        from contextlib import ExitStack

        from geosave.geodata.io.stac import open_scene
        from geosave.geodata.stac.item import read_instant
        from geosave.geodata.stac.item import build_stac_metadata
        from geosave.geodata.transform.temporal import concat

        # 1. Each collection's Items, in date order, timeless last
        items_by_collection: dict[str, list[pystac.Item]] = {}
        for entry in self.to_items():
            if entry.collection_id is None:
                raise ValueError(
                    f"Item {entry.id!r} needs a collection to name its group"
                )
            items_by_collection.setdefault(entry.collection_id, []).append(entry)

        # 2. One scene tree per row; a collection's scenes along time, node by node
        collections = self.collections
        with ExitStack() as opened:
            nodes: dict[str, xr.Dataset] = {}
            for collection_id, items in items_by_collection.items():
                dated = []
                timeless = []
                for entry in items:
                    instant = read_instant(entry)
                    if instant is None:
                        timeless.append(entry)
                    else:
                        dated.append((instant, entry))
                dated.sort(key=lambda pair: pair[0])
                dated_items = []
                for _, entry in dated:
                    dated_items.append(entry)
                dated_items.extend(timeless)
                scenes = []
                for entry in dated_items:
                    scene = open_scene(entry, **options)
                    opened.callback(scene.close)
                    scenes.append(scene)
                series = concat(scenes) if len(scenes) > 1 else scenes[0]
                metadata = build_stac_metadata(
                    dated_items, collections.get(collection_id)
                )
                # Each collection node states its own Collection.
                nodes[collection_id] = xr.Dataset(attrs=metadata.to_attrs())
                for name, raster in series.gs.rasters.items():
                    nodes[f"{collection_id}/{name}"] = raster

            tree = xr.DataTree.from_dict(nodes)
            tree.set_close(opened.pop_all().close)
            return cast("DataTree", tree)

    def to_items(self) -> list[pystac.Item]:
        """Build the pystac Item each row states, in row order.

        Returns:
            One new Item per row, holding only the assets its row has.

        Examples:
            >>> [item.id for item in rows.iloc[[0]].gs.to_items()]
            ['s0']
        """
        self._require_items()
        from geosave.geodata.stac.catalog import to_items

        return to_items(self._data)

    def _require_items(self) -> None:
        """Require the columns used to reconstruct STAC Items."""
        from geosave.geodata.stac.catalog import require_item_columns

        require_item_columns(self._data.columns)

    def _require_features(self) -> None:
        """Prevent ordinary vector writers from discarding Item metadata."""
        if FOOTER_KEY in self._data.attrs or all(
            column in self._data for column in ITEM_COLUMNS
        ):
            raise ValueError(
                "write catalog rows with to_geoparquet to preserve STAC metadata"
            )
