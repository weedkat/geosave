"""Search parameters for one STAC catalog query."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Any, Literal, Self, TypedDict

from cql2 import Expr

from geosave.geodata.core.region import GeoRegion
from geosave.geodata.utils.crs import validate_wgs84_bbox


class SortField(TypedDict):
    """One STAC sort field and its direction."""

    field: str
    direction: Literal["asc", "desc"]


class SearchParams(TypedDict, total=False):
    """Parameters shared by STAC API and GeoParquet searches."""

    collections: list[str]
    ids: list[str]
    bbox: tuple[float, float, float, float]
    intersects: dict[str, Any]
    datetime: dt | str | tuple[dt, dt]
    filter: dict[str, Any]
    filter_lang: Literal["cql2-json"]
    sortby: list[SortField]
    max_items: int
    limit: int


@dataclass(frozen=True)
class StacQuery:
    """Parameters for a single STAC search.

    Args:
        collections: Collection IDs to search. At least one.
        ids: Item IDs to select. At least one when given. Explicit spatial and
            temporal selectors are retained.
        bbox: WGS84 bounding box `(min_lon, min_lat, max_lon, max_lat)`.
        intersects: GeoJSON geometry to intersect.
        datetime: Instant, ISO 8601 string, or `(start, end)` range.
        filter: CQL2-JSON filter. Build it with `set_filter`.
        sortby: Sort keys in STAC POST form, each
            `{"field": <path>, "direction": "asc" | "desc"}`. Build it with
            `sort_by`. The API orders the items; the Planetary Computer does
            not do so reliably, so sort the item table where order matters.
        max_items: Client-side cap on items returned.
        limit: Server page-size hint.

    Raises:
        ValueError: `collections` is empty, `ids` is given but empty, `bbox`
            is invalid or combined with `intersects`, or a limit is below one.
    """

    collections: list[str]
    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, Any] | None = None
    datetime: dt | str | tuple[dt, dt] | None = None
    filter: dict[str, Any] | None = None
    max_items: int | None = None
    limit: int | None = None
    ids: list[str] | None = None
    sortby: list[SortField] | None = None

    def __post_init__(self) -> None:
        """Check the collections, bounds, and limits.

        Raises:
            ValueError: A field is empty, malformed, or below one.
        """
        validate_wgs84_bbox(self.bbox)
        if self.bbox is not None and self.intersects is not None:
            raise ValueError("bbox and intersects are mutually exclusive")
        if not self.collections:
            raise ValueError("StacQuery needs at least one collection")
        if self.ids is not None and not self.ids:
            raise ValueError("ids is set but empty; pass at least one item ID or None")
        if self.max_items is not None and self.max_items < 1:
            raise ValueError(f"max_items must be positive, got {self.max_items}")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"limit must be positive, got {self.limit}")

    def to_search_params(self) -> SearchParams:
        """Build the keyword arguments pystac-client's search takes.

        Returns:
            Search parameters, unset ones omitted.
        """
        params: SearchParams = {"collections": self.collections}
        if self.ids is not None:
            params["ids"] = self.ids
        if self.bbox is not None:
            params["bbox"] = self.bbox
        if self.intersects is not None:
            params["intersects"] = self.intersects
        if self.datetime is not None:
            params["datetime"] = self.datetime
        if self.filter is not None:
            params["filter"] = self.filter
            params["filter_lang"] = "cql2-json"
        if self.sortby is not None:
            params["sortby"] = self.sortby
        if self.max_items is not None:
            params["max_items"] = self.max_items
        if self.limit is not None:
            params["limit"] = self.limit
        return params

    def set_region(self, region: GeoRegion) -> Self:
        """Replace spatial and temporal selectors with the region's coverage.

        A timeless region clears datetime. Other criteria, including IDs,
        collections, filters and sorting, remain unchanged. This query is unchanged.
        """
        return dataclasses.replace(
            self,
            bbox=None,
            intersects=region.extent.to_crs("EPSG:4326").json,
            datetime=region.timespan,
        )

    def set_filter(self, expr: str) -> Self:
        """Add a CQL2 text expression to the filter.

        An existing filter is kept and joined with `and`.

        Args:
            expr: CQL2 text, e.g. `"eo:cloud_cover <= 10"`.

        Returns:
            New query carrying the merged filter.

        Raises:
            ValueError: `expr` is not valid CQL2 text.

        Examples:
            >>> query = StacQuery(collections=["sentinel-2-l2a"]).set_filter("eo:cloud_cover <= 10")
        """
        parsed = Expr(expr).to_json()
        merged = (
            parsed
            if self.filter is None
            else {"op": "and", "args": [self.filter, parsed]}
        )
        return dataclasses.replace(self, filter=merged)

    def sort_by(self, *fields: str) -> Self:
        """Set the sort order, replacing any order already set.

        Args:
            fields: Field paths in priority order, each optionally prefixed
                `-` for descending or `+` for ascending. Property fields need
                the `properties.` prefix, e.g. `"-properties.eo:cloud_cover"`,
                `"+id"`.

        Returns:
            New query carrying the sort order.

        Raises:
            ValueError: No fields given, or a field name is empty after the
                sign.

        Examples:
            >>> query = StacQuery(collections=["sentinel-2-l2a"]).sort_by(
            ...     "-properties.eo:cloud_cover"
            ... )
        """
        if not fields:
            raise ValueError("sort_by needs at least one field")
        keys: list[SortField] = []
        for field in fields:
            signed = field[:1] in ("+", "-")
            path = field[1:] if signed else field
            if not path:
                raise ValueError(f"sort field {field!r} has no name after the sign")
            direction = "desc" if field[:1] == "-" else "asc"
            keys.append({"field": path, "direction": direction})
        return dataclasses.replace(self, sortby=keys)
