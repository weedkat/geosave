"""Collection metadata shared by the Item and sample table accessors."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .vector import GeoVector

# Internal base; callers use GeoCatalog or GeoSamples through gs.
__all__: list[str] = []

if TYPE_CHECKING:
    import geopandas as gpd
    import pystac
    from upath import UPath

    from geosave.geodata import GeoDataFrame


class CollectionTable(GeoVector):
    """Own Collection definitions and lossless table concatenation."""

    @property
    def collections(self) -> dict[str, pystac.Collection]:
        """Return native copies of the Collection definitions attached to this table."""
        import pystac

        return {
            key: pystac.Collection.from_dict(fields)
            for key, fields in self._data.attrs.get("stac:collections", {}).items()
        }

    @classmethod
    def concat(cls, vectors: Iterable[gpd.GeoDataFrame]) -> GeoDataFrame:
        """Combine rows and merge the Collection definitions of every table.

        Args:
            vectors: Tables of the same kind and CRS, in concatenation order.

        Returns:
            Table with a fresh RangeIndex, canonical geometry column and merged
            Collection definitions. Duplicate rows are retained; for a repeated
            Collection id the first definition is kept.

        Raises:
            ValueError: No tables are supplied or their kinds or CRSs differ.

        Examples:
            >>> combined = GeoSamples.concat([training, validation])
            >>> sorted(combined.gs.collections)
            ['label', 's2']
        """
        frames = list(vectors)
        result = super().concat(frames)

        definitions = {}
        for frame in frames:
            for collection_id, fields in frame.attrs.get(
                "stac:collections", {}
            ).items():
                definitions.setdefault(collection_id, fields)

        result.attrs["stac:collections"] = deepcopy(definitions)
        return result

    def to_geojson(self, *args: Any, **kwargs: Any) -> UPath:
        """Refuse GeoJSON, which loses Collection metadata.

        Raises:
            ValueError: Always; use GeoParquet with `to_geoparquet`.
        """
        raise ValueError("Collection metadata requires GeoParquet with to_geoparquet")

    def to_geopackage(self, *args: Any, **kwargs: Any) -> UPath:
        """Refuse GeoPackage, which loses Collection metadata.

        Raises:
            ValueError: Always; use GeoParquet with `to_geoparquet`.
        """
        raise ValueError("Collection metadata requires GeoParquet with to_geoparquet")
