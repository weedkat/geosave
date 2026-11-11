"""Explore and sample spatial catalog records on native GeoDataFrames."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import Geometry

from geosave.geodata.errors import VectorValidationError
from geosave.geodata.stac.catalog import FOOTER_KEY, ITEM_COLUMNS


if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox
    import holoviews as hv
    import pystac
    from numpy.typing import DTypeLike
    from typing_extensions import Unpack
    from upath import UPath
    from geosave.geodata import DataTree, Dataset
    from geosave.geodata.io.storage import Location
    from geosave.geodata.io.formats.geojson import GeoJSONWriteOptions
    from geosave.geodata.io.formats.geopackage import GeoPackageWriteOptions
    from geosave.geodata.io.formats.geoparquet import GeoParquetWriteOptions

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core.region import GeoRegion
    from geosave.geodata.viz.explore import Explorer

    from .samples import GeoSamples


class GeoCatalog:
    """Place, draw, write and sample spatial records, including STAC Items.

    Native GeoPandas masks select rows. Spatial operations request the geometry
    and CRS guarantees they need through validate. The region combines geometry
    and temporal coverage; sampling groups source rows without opening pixels.

    Item conversion and raster loading additionally require STAC columns.
    Collection definitions travel with rows in the stac-geoparquet attrs;
    ordinary features need neither Item columns nor Collection metadata.

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

    def validate(
        self,
        *,
        geometry: bool = True,
        crs: bool = False,
        stac_item: bool | None = None,
        area: bool = False,
        temporal: bool = False,
    ) -> None:
        """Check requested table structure without changing rows or opening assets.

        Args:
            geometry: Require present, nonempty, valid geometries in every row.
                Empty tables pass; operations needing rows check that separately.
            crs: Require a declared coordinate reference system.
            stac_item: True requires STAC Item columns. False rejects STAC
                columns or metadata to require ordinary features. None skips
                this check. Does not validate full schemas or asset access.
            area: Require valid geometries with positive area, even when
                geometry=False. Points and lines cannot found spatial samples.
            temporal: Require paired, ordered temporal bounds and a nominal
                datetime within coverage. Undated rows remain valid.

        Raises:
            VectorValidationError: A requested check fails. Geometry errors
                identify the row position, independent of index labels or IDs.

        Examples:
            >>> features.gs.validate(crs=True)
            >>> items.gs.validate(geometry=False, crs=True, stac_item=True)
        """
        from geosave.geodata.core.vector.utils.geometry import validate_geometry
        from geosave.geodata.core.vector.utils.time import validate_time

        if geometry or area:
            for position, shape in enumerate(self._data.geometry):
                try:
                    validate_geometry(shape, area=area)
                except VectorValidationError as error:
                    identity = self._data.iloc[position].get("id", position)
                    raise VectorValidationError(
                        f"Row at position {position} (Item {identity!r}): {error}"
                    ) from error

        if crs and self._data.crs is None:
            raise VectorValidationError(
                "The vector has no CRS. Assign the known coordinate reference "
                "system with frame.set_crs(...); this does not reproject geometries."
            )

        if stac_item is True:
            missing = set(ITEM_COLUMNS) - set(self._data.columns)
            if missing:
                raise VectorValidationError(
                    f"STAC Item operations need columns {sorted(missing)}; item tables come "
                    "from to_catalog, StacClient.search, or reading a stac-geoparquet file"
                )

        if stac_item is False:
            if FOOTER_KEY in self._data.attrs or all(
                column in self._data.columns for column in ITEM_COLUMNS
            ):
                raise VectorValidationError(
                    "ordinary features required; write catalog rows with "
                    "to_geoparquet to preserve STAC metadata"
                )

        if temporal:
            validate_time(self._data)

    @property
    def crs(self) -> OdcCRS:
        """Return the coordinate reference system every geometry shares."""
        self.validate(geometry=False, crs=True)
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
            raise VectorValidationError(
                "An empty vector has no footprint. Select at least one row."
            )
        self.validate()
        return Geometry(self._data.geometry.union_all(), crs=self.crs)

    @property
    def region(self) -> GeoRegion:
        """Return feature union and the temporal envelope of dated rows.

        Timeless rows do not widen the envelope. No assets are opened and no
        pixel grid is inferred. Invalid geometry or incomplete dates raise.
        """
        from geosave.geodata.core.region import GeoRegion
        from geosave.geodata.core.vector.utils.time import read_times

        region = GeoRegion.from_geometry(self.footprint)
        self.validate(geometry=False, temporal=True)
        times = read_times(self._data)
        dated = times.loc[times["start"].notna()]
        if dated.empty:
            return region

        return region.replace(timespan=(dated["start"].min(), dated["end"].max()))

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
                holds a second column named `geometry` beside its active one,
                or STAC Item tables are mixed with ordinary features.
        """
        items = list(vectors)
        if not items:
            raise ValueError("concat needs at least one GeoDataFrame")
        stac_frames = []
        for item in items:
            stac_frames.append(
                FOOTER_KEY in item.attrs
                or all(column in item for column in ITEM_COLUMNS)
            )

        if any(stac_frames) and not all(stac_frames):
            raise ValueError(
                "cannot concatenate STAC Item tables with ordinary features"
            )

        for item in items:
            item.gs.validate(geometry=False, crs=True, stac_item=all(stac_frames))

        crs = OdcCRS(items[0].crs)
        mismatches = [str(item.crs) for item in items if OdcCRS(item.crs) != crs]
        if mismatches:
            raise ValueError(
                f"the vectors' CRSs differ from {crs}: {mismatches}; "
                "call to_crs explicitly"
            )

        # Normalize the active geometry name without overwriting a property.
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

        from geosave.geodata.stac.catalog import merge_collections

        definitions = []
        footer = {}
        for frame in items:
            definitions.extend(frame.gs.collections.values())
            for key, value in frame.attrs.get(FOOTER_KEY, {}).items():
                if key == "collections":
                    continue
                if key in footer and footer[key] != value:
                    raise ValueError(f"catalog footer metadata disagrees on {key!r}")
                footer[key] = deepcopy(value)
        if any(FOOTER_KEY in frame.attrs for frame in items):
            footer["collections"] = {}
            for key, collection in merge_collections(definitions).items():
                footer["collections"][key] = collection.to_dict(transform_hrefs=False)
            result.attrs[FOOTER_KEY] = footer
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
            >>> scene.gs.plot() * plots.to_crs(region.crs).gs.plot()
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
        """Construct a web map viewer with a sidebar to explore this vector.

        Features have hover and click identification. Property and date filters
        narrow the table and map. Rows start unselected. Selecting a row
        draws its geometry; the Raster control opens Item assets separately.

        Args:
            column: Column that colours the features, or None for one colour.
            basemap: Basemap under the layers, as `viz.explore.page.map.BASEMAPS` names it.
                None uses the light basemap.
            color: Colour of every feature when no column colours them.
            alpha: Opacity of polygon fill; 0 draws boundaries only.

        Returns:
            Explorer, shown by a notebook or started with `.serve()`.

        Raises:
            KeyError: `column` is not a column of the frame.
            ValueError: The vector has no CRS, holds no geometry, or holds a
                GeometryCollection.

        Examples:
            >>> fields.gs.explore(column="crop")
            >>> fields.gs.explore(alpha=0.0, basemap="Imagery")  # boundaries only
        """
        from matplotlib.colors import to_hex
        from geosave.geodata.viz.explore import Explorer

        viewer = Explorer(tables={"Features": self._data}, basemap=basemap)
        layer = viewer.tables.tables["Features"].footprints
        layer.column = column
        layer.color = to_hex(color)
        layer.alpha = alpha
        return viewer

    def rasterize(
        self,
        like: GeoBox | xr.DataArray | xr.Dataset,
        *,
        column: str | None = None,
        fill: int | float | bool = 0,
        dtype: DTypeLike | None = None,
        all_touched: bool = False,
    ) -> xr.DataArray:
        """Burn geometries or one property onto an existing exact grid.

        Args:
            like: GeoBox or geolocated xarray object supplying the target grid.
            column: Property to burn. None returns a boolean presence mask.
            fill: Value outside geometries. A presence mask takes only 0 or
                False, since it is False outside them.
            dtype: Output dtype for property values.
            all_touched: Burn every touched pixel instead of pixel centres.

        Returns:
            Eager geolocated array on the exact target grid.

        Raises:
            KeyError: The selected property is absent.
            ValueError: A geometry is null, empty, or invalid, values or fill
                cannot be represented by the dtype, or `column` is None with a
                truthy `fill`.
        """
        from odc.geo.geobox import GeoBox

        from geosave.geodata.conventions import SPATIAL_DIMENSIONS
        from geosave.geodata.core.vector.utils.rasterize import (
            burn_column,
            burn_presence,
        )

        if isinstance(like, GeoBox):
            if like.crs is None:
                raise ValueError("rasterize target grid carries no CRS")
            geobox = like
        else:
            like.gs.validate(georeferenced=True, affine=True)
            geobox = cast("GeoBox", like.gs.geobox)

        self.validate(crs=True)
        frame = self._data.to_crs(cast("OdcCRS", geobox.crs))
        if column is None:
            pixels = burn_presence(frame, geobox, fill, all_touched)
        else:
            pixels = burn_column(frame, geobox, column, fill, dtype, all_touched)

        raster = xr.DataArray(
            pixels, dims=SPATIAL_DIMENSIONS, name=column or "mask"
        ).gs.write_grid(geobox)
        if isinstance(like, GeoBox):
            return raster

        # Sliced coordinates can drift from regenerated affine coordinates.
        coords = {}
        for dim in SPATIAL_DIMENSIONS:
            coords[dim] = like.coords[dim]
        return raster.assign_coords(coords)

    @property
    def collections(self) -> dict[str, pystac.Collection]:
        """Return the Collections the table's footer metadata holds, by id."""
        import pystac

        footer = self._data.attrs.get(FOOTER_KEY, {})
        collections = {}
        for collection_id, fields in footer.get("collections", {}).items():
            collections[collection_id] = pystac.Collection.from_dict(fields)
        return collections

    def to_geojson(
        self,
        path: Location,
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
        from geosave.geodata.io.formats import geojson

        return geojson.write_geojson(self._data, path, overwrite=overwrite, **options)

    def to_geopackage(
        self,
        path: Location,
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
        collections: Sequence[str] | None = None,
        match: Literal["exact", "overlap"] = "overlap",
        overlap: float = 0.9,
        date_range: tuple[str | datetime | None, str | datetime | None] | None = None,
        freq: str | pd.offsets.BaseOffset | None = None,
        window: int = 1,
        step: int = 1,
        interval: Literal["raise", "first", "last", "overlap", "drop"] = "raise",
        missing: Literal["drop", "keep"] = "drop",
    ) -> GeoSamples:
        """Plan spatial samples and consecutive temporal windows without reading pixels.

        Collections populate slots independently. No collection supplies reference
        dates for another. With ``freq``, calendar slots are left-closed and
        labelled by their start. Otherwise exact observation timestamps define
        the timeline. Empty calendar slots remain between observations.

        With all temporal options at their defaults, return one raw sample per
        spatial group. Indexing returns source rows. Open them with
        ``samples[i].gs.to_datatree()``; loading retains their original dates.
        Slot labels describe membership, not rewritten observation timestamps.

        Args:
            collections: Required collection names. None selects all. Unselected
                static collections are not implicitly included.
            match: Equal footprints, or smallest-first overlap grouping.
            overlap: Required area fraction for overlap grouping.
            date_range: Inclusive lower and exclusive upper selection bounds;
                None means unbounded. The lower bound anchors relative frequencies.
                Anchored frequencies such as "MS" retain their calendar boundaries.
            freq: Pandas frequency string or offset. Relative DateOffset values
                require a lower date_range bound. None uses exact instants;
                interval-only Items then require an explicit frequency.
            window: Positive count of consecutive slots in each sample.
            step: Positive count of slots between window starts.
            interval: Assignment for interval-only Items crossing slots: raise,
                first, last, overlap (reuse in every intersected slot), or drop.
                Item endpoints are inclusive. An existing datetime takes priority
                for assignment; its full coverage is still preserved.
            missing: Drop windows lacking required observations, or retain them
                with empty slot membership. This never fills pixels or creates
                placeholder catalog rows.

        Returns:
            Samples retaining original catalog positions, slot membership and
            spatial footprints. No valid windows produces an empty result.

        Raises:
            ValueError: Options or temporal coverage are invalid, or an interval
                crosses slots with interval='raise'. Errors identify the Item.
            KeyError: A requested collection is absent.
            VectorValidationError: Geometry is missing, invalid, empty or has no area.

        Examples:
            >>> samples = items.gs.sample(
            ...     collections=["optical", "radar"], match="exact",
            ...     freq="MS", window=6, interval="raise",
            ... )
            >>> samples[0]  # original Item rows, without opening assets
            <GeoDataFrame ...>
            >>> with samples[0].gs.to_datatree() as tree:
            ...     print(tree.gs.groups)
            ('optical', 'radar')
        """
        from .samples import GeoSamples
        from .utils.time import read_times
        from .utils.sampling import (
            assign_slots,
            build_timeline,
            clip_observations,
            group_footprints,
            read_frequency,
            read_range,
            select_collections,
            select_range,
            select_windows,
            validate_options,
        )

        validate_options(
            collections=collections,
            match=match,
            overlap=overlap,
            window=window,
            step=step,
            interval=interval,
            missing=missing,
        )
        bounds = read_range(date_range)
        offset = read_frequency(freq, bounds[0])
        temporal = (
            collections is not None
            or freq is not None
            or window != 1
            or step != 1
            or interval != "raise"
            or missing != "drop"
        )

        # The working index is source position, independent of duplicate user
        # labels. Filtering and slot assignment retain it until construction.
        rows = self._data.copy()
        rows.index = pd.RangeIndex(len(rows))
        rows = select_collections(rows, collections)
        times = None
        if temporal or bounds != (None, None):
            rows.gs.validate(geometry=False, temporal=True)
            times = select_range(read_times(rows), bounds)
            rows = rows.loc[times.index]

        footprints, groups = group_footprints(rows, match, overlap)
        if not temporal:
            return GeoSamples(self._data, groups, footprints=footprints)

        names = list(collections) if collections is not None else []
        if len(rows):
            if "collection" not in rows:
                raise ValueError("Temporal sampling requires a collection column")
            if not names:
                names = list(rows["collection"].drop_duplicates())
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError("Temporal sampling requires nonempty collection names")

        # Date filtering can use plain features; only calendar membership needs
        # collection names and row identity alongside the temporal values.
        assert times is not None
        observations = times.assign(
            row=rows.index,
            id=rows.get("id", pd.Series(rows.index, index=rows.index)),
            collection=rows.get("collection", pd.Series(index=rows.index, dtype=str)),
        )
        observations = clip_observations(observations, bounds)

        indices = []
        windows = []
        shapes = []
        for shape, members in zip(footprints, groups, strict=True):
            group = observations.loc[observations.index.isin(members)]
            timeline = build_timeline(group, offset, bounds)
            slots = assign_slots(group, timeline, interval)
            frames = select_windows(
                slots,
                timeline,
                names,
                window,
                step,
                missing,
                trim=bounds == (None, None),
            )
            for frame in frames:
                positions = []
                for membership in frame["indices"]:
                    for position in membership:
                        if position not in positions:
                            positions.append(position)
                indices.append(positions)
                windows.append(frame)
                shapes.append(shape)

        return GeoSamples(
            self._data,
            indices,
            slots=windows,
            footprints=gpd.GeoSeries(shapes, crs=rows.crs),
        )

    def to_datatree(self, **options: Any) -> DataTree:
        """Load the Items lazily as ``{collection}/{asset}`` nodes on native grids.

        Each collection loads with `stac.load_datatree`: each STAC asset is a node
        on the grid it states, and the collection's time sits on its own node.
        Collections follow their first appearance. Merge assets on one grid
        with `tree.gs.group_grid()`.

        Args:
            **options: Forwarded to `stac.load_datatree`, such as `groupby`
                or `chunks`.

        Returns:
            DataTree of collection nodes, each carrying its Collection and
            extension schemas as `StacMetadata` attrs, over their asset nodes.
            Each collection's region is its Items' geometry, beside the whole
            grids.

        Raises:
            ValueError: An Item names no collection, or a band states no grid.

        Examples:
            >>> tree = items.gs.to_datatree()
            >>> tree.gs.groups
            ('s2', 'dem')
            >>> list(tree.gs.datasets)
            ['s2/B04', 's2/B11', 'dem/elevation']
            >>> list(tree.gs.group_grid().gs.datasets)
            ['s2/r10m', 's2/r20m', 'dem/r30m']
        """
        from geosave.geodata.stac.load import load_datatree
        from geosave.geodata.stac.item import read_footprint

        items_by_collection: dict[str, list[pystac.Item]] = {}
        for entry in self.to_items():
            if not entry.collection_id:
                raise ValueError(
                    f"Item {entry.id!r} needs a collection to name its node"
                )
            items_by_collection.setdefault(entry.collection_id, []).append(entry)
        collections = self.collections
        # Each collection loads on its own grids; their nodes join as siblings.
        nodes = {}
        for identity, entries in items_by_collection.items():
            tree = load_datatree(
                entries, collection=collections.get(identity), **options
            )
            # The Items' geometry is their region; the grid stays whole.
            footprint = read_footprint(entries)
            if footprint is not None:
                tree = tree.gs.write_extent(footprint)
            for path, node in tree.subtree_with_keys:
                if path != ".":
                    nodes[path] = node.to_dataset(inherit=False)
        return cast("DataTree", xr.DataTree.from_dict(nodes))

    def to_dataset(self, geobox: GeoBox, **options: Any) -> Dataset:
        """Load one collection's Items lazily onto one grid.

        Args:
            geobox: Grid every band lands on.
            **options: Forwarded to `stac.load_dataset`, such as `groupby`,
                `bands` or `chunks`.

        Returns:
            Dataset on `geobox`, carrying its Collection as `StacMetadata` attrs.

        Raises:
            ValueError: The rows hold more than one collection; load each with
                its own `to_dataset`, or all of them with `to_datatree`.

        Examples:
            >>> scene = items.gs.to_dataset(grid)
            >>> scene.gs.geobox == grid
            True
        """
        from geosave.geodata.stac.load import load_dataset

        entries = self.to_items()
        identities = list(dict.fromkeys(entry.collection_id for entry in entries))
        if len(identities) > 1:
            raise ValueError(
                f"to_dataset loads one collection, the rows hold {identities}"
            )
        collection = self.collections.get(identities[0]) if identities else None
        return cast(
            "Dataset", load_dataset(entries, geobox, collection=collection, **options)
        )

    def to_items(self) -> list[pystac.Item]:
        """Build the pystac Item each row states, in row order.

        Returns:
            One new Item per row, holding only the assets its row has.

        Examples:
            >>> [item.id for item in rows.iloc[[0]].gs.to_items()]
            ['s0']
        """
        self.validate(geometry=False, crs=True, stac_item=True)
        from geosave.geodata.stac.catalog import to_items

        return to_items(self._data)
