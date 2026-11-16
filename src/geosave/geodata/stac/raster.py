"""STAC hierarchy with opened xarray raster assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, Unpack

import pystac
import xarray as xr

from geosave.geodata.core.conventions import (
    END_DATETIME,
    ITEM_ID_FORMAT,
    START_DATETIME,
    TIME_COORDINATE,
)
from geosave.geodata.io import storage
from geosave.geodata.stac.publish import StacItem

if TYPE_CHECKING:
    from geosave.geodata.io.raster.geotiff import COGWriteOptions
    from geosave.geodata.io.storage import Location, StorageOptions


@dataclass(eq=False)
class RasterStac:
    """Represent one collection's raster Items as an xarray tree.

    ``collection`` holds the native pystac.Collection, when available.
    ``tree`` holds Items with scalar coordinates and assets on their own grids
    (10 m B04 and 20 m SCL in this small example)::

        >>> raster_stac.tree
        <xarray.DataTree>
        Group: /
        └── Group: /20250601T103000
            │   <xarray.Dataset>     Item metadata; no raster variables
            │   Dimensions:         ()
            │   Coordinates:
            │       time            datetime64[ns] 8B 2025-06-01T10:30:00
            │       start_datetime  datetime64[ns] 8B 2025-06-01T10:30:00
            │       end_datetime    datetime64[ns] 8B 2025-06-01T10:35:00
            │       eo_cloud_cover  float64 8B 12.5
            │       platform        <U11 44B 'sentinel-2b'
            │   Data variables:
            │       *empty*
            │   Attributes:
            │       stac_extensions: [schema URIs for the Item properties]
            ├── Group: /20250601T103000/B04
            │       Dimensions:      (y: 2, x: 4)
            │       Coordinates:
            │         * y            (y) float64 16B 5e+06 5e+06
            │         * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
            │           spatial_ref  int32 4B 32633
            │       Data variables:
            │           B04          (y, x) uint16 16B dask.array<chunksize=(2, 4), meta=np.ndarray>
            └── Group: /20250601T103000/SCL
                    Dimensions:      (y: 1, x: 2)
                    Coordinates:
                      * y            (y) float64 8B 5e+06
                      * x            (x) float64 16B 5e+05 5e+05
                        spatial_ref  int32 4B 32633
                    Data variables:
                        SCL          (y, x) uint8 2B dask.array<chunksize=(1, 2), meta=np.ndarray>

    Time and marked property coordinates live on Item nodes and inherit into
    assets. Extension declarations stay in Item attrs; Collection metadata
    lives on the wrapper. Spatial coordinates and band metadata stay on assets.
    In native DataTree, attrs do not inherit. Property marks and band metadata
    remain on their variables::

        >>> raster_stac.tree["20250601T103000"]["eo_cloud_cover"].attrs
        {'stac_property': 'eo:cloud_cover'}
        >>> raster_stac.tree["20250601T103000/B04"]["B04"].attrs
        {'scale_factor': 0.0001, 'add_offset': 0.0, '_FillValue': 0,
         'long_name': 'Red reflectance', 'units': '1',
         'common_name': 'red', 'center_wavelength': 0.665}
        >>> raster_stac.tree["20250601T103000/SCL"]["SCL"].attrs
        {'_FillValue': 0, 'flag_values': [4, 5, 6],
         'flag_meanings': 'vegetation bare_soil water'}

    The tree stages a publication; it is not a loading format. ``from_dataset``
    splits a conventional raster without computing pixels, ``merge`` joins
    rasters on other grids into the same Items, and ``to_cog`` writes the
    files and returns their Items. Published Items load back as Datasets with
    ``stac.load_dataset`` or ``stac.load_native_datasets``. Keep file-backed
    assets open while using the tree.

    Args:
        tree: Item/asset hierarchy with raster Datasets at asset nodes.
        collection: Native Collection metadata, or None when unavailable.
    """

    tree: xr.DataTree
    collection: pystac.Collection | None = None

    @classmethod
    def from_dataset(
        cls,
        dataset: xr.Dataset,
        *,
        collection: pystac.Collection | None = None,
        item_dim: str = TIME_COORDINATE,
        id_format: str = ITEM_ID_FORMAT,
    ) -> Self:
        """Group by Item dimension, then split each group into asset Datasets.

        Uses native xarray grouping and variable selection. Time and marked
        properties become Item coordinates. Dataset attrs become Item attrs,
        except stac_collection, which belongs to the wrapper. Spatial coordinates, variable attrs, and encoding stay on assets.
        Each asset wraps its DataArray as a Dataset, without source Dataset attrs.

        Args:
            dataset: Source raster Dataset; keep it open while using lazy assets.
            collection: Collection the Items belong to. None keeps the one the
                Dataset states, as `stac.load_dataset` stamps it; give one for
                a derived raster, which is no longer its source's Collection.
            item_dim: Dimension to group by. Each group must have one entry.
                The dimension must exist on the Dataset.
            id_format: GeoRegion.format template for each group's Item id.

        Returns:
            Item/asset tree with one variable per asset and source Collection
            metadata on the wrapper. Pixels are not computed.
        """
        # item_dim controls grouping; time coordinates describe the observation.
        time_coords = (TIME_COORDINATE, START_DATETIME, END_DATETIME)
        property_coords = dataset.gs.stac_properties.values()
        item_coord = [
            name for name in dataset.coords if name in time_coords or name in property_coords
        ]

        item_attrs = dict(dataset.attrs)
        item_attrs.pop("stac_collection", None)

        nodes: dict[str, xr.Dataset] = {}
        for _, group in dataset.groupby(item_dim):
            observation = group.squeeze(item_dim)
            item_id = observation.gs.region.format(id_format)
            if item_id in nodes:
                raise ValueError(f"Duplicate Item id {item_id!r}")

            item_coords = {
                name: observation.coords[name].variable for name in item_coord
            }
            nodes[item_id] = xr.Dataset(coords=item_coords, attrs=item_attrs)

            assets = observation.drop_vars(item_coord)
            for name in assets.data_vars:
                nodes[f"{item_id}/{name}"] = assets[name].to_dataset()

        tree = xr.DataTree.from_dict(nodes)
        if collection is None:
            collection = dataset.gs.stac_collection
        return cls(tree=tree, collection=collection)

    def merge(self, other: RasterStac) -> RasterStac:
        """Join another staged hierarchy: new assets into shared Items, and new Items.

        Items match by id. A raster on another grid, staged with the same
        `id_format`, adds its assets to the Items both state; Items only one
        states are kept as they are.

        Args:
            other: Hierarchy of the same Collection.

        Returns:
            New hierarchy with every Item and asset of both; pixels are not
            computed.

        Raises:
            ValueError: Both state an asset, state an Item differently (its
                time, properties or attrs), or state different Collections.

        Examples:
            >>> fine = RasterStac.from_dataset(bands_10m)
            >>> staged = fine.merge(RasterStac.from_dataset(bands_20m))
            >>> list(staged.tree["20250601T103000"].children)
            ['B04', 'B08', 'SCL']
        """
        collection = self.collection
        if collection is None:
            collection = other.collection
        elif other.collection is not None:
            if collection.to_dict() != other.collection.to_dict():
                raise ValueError(
                    f"Collections differ: {collection.id!r} and {other.collection.id!r}"
                )

        datasets_by_path: dict[str, xr.Dataset] = {}
        for node in self.tree.subtree:
            datasets_by_path[node.path] = node.to_dataset(inherit=False)

        for other_node in other.tree.subtree:
            other_dataset = other_node.to_dataset(inherit=False)
            path = other_node.path
            if path not in datasets_by_path:
                datasets_by_path[path] = other_dataset
                continue
            # Only asset nodes hold variables; a shared Item node must agree.
            if other_dataset.data_vars:
                raise ValueError(f"Asset {path!r} is in both")
            if not other_dataset.identical(datasets_by_path[path]):
                raise ValueError(f"Item {path!r} differs between the two")

        merged_tree = xr.DataTree.from_dict(datasets_by_path)
        return RasterStac(tree=merged_tree, collection=collection)

    def to_cog(
        self,
        root: Location,
        *,
        overwrite: bool = False,
        storage_options: StorageOptions | None = None,
        **options: Unpack[COGWriteOptions],
    ) -> list[StacItem]:
        """Write each asset as a COG and return the Items that state them.

        Files land at `<root>/<item id>/<asset key>.tif`, and each Item states
        its assets by absolute href and links to the Collection, when there
        is one. No Item or Collection JSON is written: `stac.create_catalog`
        lays the Items out as a table, which stores the Collection.

        Args:
            root: Folder or URL prefix to publish under.
            overwrite: Replace existing files when true.
            storage_options: Options for the filesystem a URL names.
            **options: COG creation options, passed to GDAL unchanged.

        Returns:
            One Item per Item node, in tree order, covering the node's region:
            the union of its assets' grids over its timespan.

        Raises:
            FileExistsError: A file exists and `overwrite` is false.
            RasterValidationError: An Item node has no assets.
            ValueError: An Item node states no time.

        Examples:
            >>> items = RasterStac.from_dataset(scene).to_cog("s3://bucket/s2")
            >>> items[0].assets["B04"].href
            's3://bucket/s2/20250601T103000/B04.tif'
            >>> catalog = stac.create_catalog(items)
        """
        from geosave.geodata.io.raster import geotiff

        root = storage.create_upath(root, storage_options)

        items = []
        for item_id, item_node in self.tree.children.items():
            item_coords = item_node.to_dataset(inherit=False)
            item = StacItem.from_region(
                item_node.gs.region,
                id=item_id,
                properties=item_coords.gs.read_stac_properties(),
            )
            item.stac_extensions = item_coords.gs.stac_extensions
            if self.collection is not None:
                item.set_collection(self.collection)

            for asset_key, asset_node in item_node.children.items():
                asset_raster = asset_node.to_dataset(inherit=False)
                asset_path = root / item_id / f"{asset_key}.tif"
                geotiff.write_cog(asset_raster, asset_path, overwrite=overwrite, **options)
                item.add_raster_asset(asset_key, asset_raster, str(asset_path.absolute()))
            items.append(item)
        return items
