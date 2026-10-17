"""The `gs` accessor of an item table: one row per STAC Item.

An item table is where every sample starts: `sample` turns it into a sample
table, read through `GeoSamples`. The table also writes itself as STAC.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any


from .table import CollectionTable

if TYPE_CHECKING:
    import pystac
    from upath import UPath

    from geosave.geodata import GeoDataFrame
    from .samples import GeoSamples
    from geosave.geodata.io.storage import Location, StorageOptions


class GeoCatalog(CollectionTable):
    """GeoSave operations on an item table, read through `gs`.

    Each row is one STAC Item, read through `GeoItem`. The Collection
    definitions its rows name sit in `attrs["stac:collections"]` in memory and
    in the file's `stac-geoparquet` metadata on disk. ``sample`` turns the Items
    into samples.

    Examples:
        >>> items = gs.open_vector("data/catalog.parquet")
        >>> items.iloc[0].gs.to_xarray()
        <xarray.Dataset> ...
        >>> samples = items.gs.sample()
    """

    # Columns every STAC Item row carries; their presence makes a frame an item table.
    COLUMNS = ("type", "stac_version", "id", "assets")

    def sample(self) -> GeoDataFrame[GeoSamples]:
        """Group Items sharing a footprint, independently of grids and dates.

        Footprints are compared in WGS84 after normalizing vertex order.
        Each row retains every Item, grouped by collection and ordered by
        acquisition date. Containing or partly overlapping footprints remain
        separate samples. No files are opened.

        Returns:
            Gridless sample table, one row per footprint, with Collection metadata.

        Raises:
            ValueError: Items lack a CRS, collection id or valid footprint.

        Examples:
            >>> samples = items.gs.sample()
            >>> samples.iloc[0].gs.to_xarray()
            <xarray.DataTree> ...
        """
        from geosave.geodata.sampling.sample import sample

        return sample(self._data)

    def to_geoparquet(
        self,
        path: Location,
        *,
        collections: Iterable[pystac.Collection] | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Any,
    ) -> UPath:
        """Write this item table as one stac-geoparquet file.

        Asset hrefs are stored relative to the file, so a table and its assets
        move together. One file holds every Collection its rows name.

        Args:
            path: Output path or URL ending in `.parquet` or `.geoparquet`.
            collections: Definitions overriding attached Collections by id.
                Referenced definitions are required and copied with updated extents.
            overwrite: Replace an existing file when true.
            storage_options: Options for the filesystem a URL names.
            **options: Not accepted; stac-geoparquet fixes its own encoding.

        Returns:
            Local writes return a PosixUPath; URL writes return a UPath
            carrying its storage options.

        Raises:
            FileExistsError: The file exists and `overwrite` is false.
            ValueError: The suffix is wrong, a referenced Collection definition is missing,
                or options are passed.

        Examples:
            >>> rows.gs.to_geoparquet("data/catalog.parquet", collections=[forest])
            PosixUPath('data/catalog.parquet')
        """
        if options:
            raise ValueError(
                f"stac-geoparquet fixes its own encoding; {sorted(options)} do not apply"
            )
        # Deferred: only an item table needs the STAC dependencies.
        from geosave.geodata.io.vector import stac_geoparquet

        return stac_geoparquet.write(
            self._data,
            path,
            collections=collections,
            overwrite=overwrite,
            storage_options=storage_options,
        )

    def to_items(self) -> list[pystac.Item]:
        """Build the pystac Item each row states, in row order.

        Returns:
            One new Item per row, holding only the assets its row has.

        Examples:
            >>> [item.id for item in rows.iloc[[0]].gs.to_items()]
            ['s0']
        """
        from geosave.geodata.stac.translate import table as stac_table

        return stac_table.to_items(self._data)

    def to_stac_json(
        self,
        path: Location,
        *,
        collections: Iterable[pystac.Collection] | None = None,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
    ) -> UPath:
        """Write this item table as a self-contained static STAC catalog.

        The folder holds `catalog.json`, one `<collection>/collection.json` per
        Collection and one `<collection>/<id>/<id>.json` per Item, asset hrefs
        relative so the folder and its assets move together.

        Args:
            path: Folder to write into, as a path, URL, or UPath.
            collections: Definitions overriding attached Collections by id.
                Referenced definitions are required and copied with updated extents.
            overwrite: Replace a catalog already in the folder when true.
            storage_options: Options for the filesystem a URL names.

        Returns:
            The root `catalog.json`: a `PosixUPath` locally, else the remote
            UPath carrying its storage options.

        Raises:
            FileExistsError: The folder holds a catalog and `overwrite` is false.
            ValueError: An id repeats within one collection, or a referenced
                Collection definition is missing.

        Examples:
            >>> rows.gs.to_stac_json("data/stac", collections=[forest])
            PosixUPath('data/stac/catalog.json')
            >>> root = pystac.Catalog.from_file("data/stac/catalog.json")
            >>> gs.stac.create_catalog(root.get_items(recursive=True))
        """
        from geosave.geodata.io.vector import stac_json

        return stac_json.write(
            self.to_items(),
            path,
            collections=(
                {**self.collections, **{c.id: c for c in collections or ()}}
            ).values(),
            overwrite=overwrite,
            storage_options=storage_options,
        )
