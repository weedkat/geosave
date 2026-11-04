"""Model-owned STAC search and raster loading recipe."""

from __future__ import annotations

from typing import Any, Annotated, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    PositiveInt,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from pystac_client.exceptions import APIError
import xarray as xr

from geosave.geodata.core import GeoAnchor
from geosave.geodata.errors import CollectionNotFoundError
from geosave.geodata.stac.search.api import StacClient, StacProvider
from geosave.geodata.stac.search.query import StacQuery
from collections.abc import Callable
from dataclasses import replace
from odc.geo.geobox import GeoBox
from odc.stac.model import PropertyLoadRequest
from pydantic import BaseModel
from geosave.geodata.io.odc import load_dataset
from geosave.geodata.transform.warp import Resampling

from .base import SpecModel, Text


Bands = list[str] | tuple[str, ...]
StacGroupby = Literal["solar_day", "id", "time"]
ChunkSize = int | Literal["auto"]


def _default_chunks() -> dict[str, ChunkSize]:
    """Return the default spatial Dask chunks."""
    return {"x": 1024, "y": 1024}


class StacLoadConfig(BaseModel):
    """How one source loads its pixels.

    Attributes:
        bands: Band names to load. None loads whatever the matched items
            publish.
        groupby: How odc-stac groups scenes along the time axis.
        chunks: Dask chunk sizes for the spatial dims. None loads eagerly.
        resampling: Resampling odc-stac applies, one mode or one per band.
            None keeps each asset's default.
        dtype: Output dtype. None keeps each collection's own.
        nodata: Nodata value, paired with `dtype`. None keeps each asset's own.
        fail_on_error: False skips a scene that fails to read instead of
            raising.
        with_properties: Item properties that differ between dates, recorded
            on the `stac_properties` coordinate: property keys, or odc-stac
            request mappings with `key` and optional `dtype` (`"str"` for
            text), `nodata` and `fuser`, e.g. `("view:sun_azimuth",
            "eo:cloud_cover")`. A supplied `name` must equal `key`.
        stac_cfg: Per-collection band overrides odc-stac applies, correcting
            nodata, dtype, or asset aliases a catalog publishes wrongly.
        pool: Reader threads odc-stac uses. None takes its default.
        progress: Progress bar odc-stac reports reads through. None is silent.
        patch_url: Called on every asset href before it is read, for a catalog
            whose URLs need signing at load time.
        preserve_original_order: True keeps the matched items in search order
            rather than sorting them by time.

    Examples:
        >>> StacLoadConfig(bands=["B04", "B08"], with_properties=["eo:cloud_cover"])
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    bands: Bands | None = None
    groupby: StacGroupby = "solar_day"
    chunks: dict[str, ChunkSize] | None = Field(default_factory=_default_chunks)
    resampling: Resampling | dict[str, Resampling] | None = None
    dtype: str | None = None
    nodata: float | None = None
    fail_on_error: bool = True
    with_properties: tuple[str | dict[str, Any], ...] | None = None
    stac_cfg: dict[str, Any] | None = None
    pool: int | None = None
    progress: Any | None = None
    patch_url: Callable[[str], str] | None = None
    preserve_original_order: bool = False

    @model_validator(mode="after")
    def exact_property_names(self) -> Self:
        """Keep STAC property identity in coordinate names."""
        for request in PropertyLoadRequest.from_user_input(self.with_properties or ()):
            if request.name is not None and request.name != request.key:
                raise ValueError(
                    "with_properties must keep the exact STAC key as its name"
                )
        return self

    def to_load_kwargs(self) -> dict[str, Any]:
        """Return native odc-stac settings.

        Returns:
            New keyword mapping for `odc.stac.load`, omitting unset options
            except `chunks=None`, which explicitly requests eager loading.
            Property requests are emitted as a list.
        """
        options = self.model_dump(exclude_none=True)
        # None explicitly selects eager loading; ODC otherwise chooses its default.
        options["chunks"] = self.chunks
        if self.with_properties is not None:
            requests = []
            for request in self.with_properties:
                fields = {"key": request} if isinstance(request, str) else dict(request)
                # ODC sanitizes names by default; its native name option preserves the key.
                fields["name"] = fields["key"]
                requests.append(fields)
            options["with_properties"] = requests
        return options


class SortConfig(SpecModel):
    """Declare one STAC sort field and direction.

    Attributes:
        field: Item property used for sorting, such as `datetime`.
        direction: Ascending (`asc`) or descending (`desc`) order.
    """

    field: Text
    direction: Literal["asc", "desc"]


class QueryConfig(SpecModel):
    """Declare the STAC item search owned by a model raster.

    Attributes:
        bbox: Search bounds `(west, south, east, north)` in longitude/latitude.
        intersects: GeoJSON geometry the items must intersect.
        datetime: Acquisition instant or interval accepted by the STAC query.
        filter: CQL2 JSON filter expression.
        max_items: Maximum items collected across response pages.
        limit: Requested items per response page.
        ids: Item identifiers to select.
        sortby: Ordered property/direction declarations.
    """

    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, JsonValue] | None = None
    datetime: str | tuple[str, str] | None = None
    filter: dict[str, JsonValue] | None = None
    max_items: PositiveInt | None = None
    limit: PositiveInt | None = None
    ids: tuple[Text, ...] | None = None
    sortby: tuple[SortConfig, ...] | None = None

    def to_query(self, collection: str) -> StacQuery:
        """Build the native query with its collection identity.

        Args:
            collection: Collection identifier to search.

        Returns:
            Validated `StacQuery` containing the declared non-null settings.
            This does not connect to a catalog or retrieve items.
        """
        values = self.model_dump(exclude_none=True)
        if "sortby" in values:
            values["sortby"] = [dict(item) for item in values["sortby"]]
        return StacQuery(collections=[collection], **values)

    @model_validator(mode="after")
    def _validate_query(self) -> Self:
        # StacQuery owns the search rules, so building one checks them.
        self.to_query("validation")
        return self


class StacRecipe(SpecModel):
    """State how to find and load one raster from STAC.

    Attributes:
        collection: STAC collection identifier.
        endpoints: Built-in provider names or HTTP(S) catalog endpoints in
            fallback order.
        query: Item search parameters.
        load: Native raster loading parameters.
    """

    collection: Text
    endpoints: Annotated[tuple[StacProvider | HttpUrl, ...], Field(min_length=1)]
    query: QueryConfig = Field(default_factory=QueryConfig)
    load: StacLoadConfig = Field(default_factory=StacLoadConfig)

    @field_serializer("endpoints")
    def _serialize_endpoints(self, endpoints: tuple[HttpUrl, ...]) -> tuple[str, ...]:
        return tuple(str(url) for url in endpoints)

    @field_validator("load", mode="before")
    @classmethod
    def _validate_primitive_load(cls, value: Any) -> Any:
        if isinstance(value, StacLoadConfig):
            value = value.model_dump()

        def lists(item: Any) -> Any:
            if isinstance(item, tuple):
                return [lists(member) for member in item]
            if isinstance(item, dict):
                return {key: lists(member) for key, member in item.items()}
            return item

        return TypeAdapter(
            dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False)
        ).validate_python(lists(value), strict=True)

    @field_serializer("load")
    def _serialize_load(self, load: StacLoadConfig) -> dict[str, JsonValue]:
        """Serialize native load settings as portable primitive values."""
        return load.model_dump(mode="json")

    @model_validator(mode="after")
    def _validate_endpoints(self) -> Self:
        endpoints = tuple(map(str, self.endpoints))
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("endpoints must not contain duplicate URLs")
        return self

    def load_raster(self, anchor: GeoAnchor, /, *, geobox: GeoBox) -> xr.Dataset:
        """Load a lazy raster on an anchor from the first endpoint serving it.

        An endpoint that cannot be reached, or does not publish the
        collection, passes to the next; a failure while loading does not.

        Args:
            anchor: Geographic extent and datetime window to search.
            geobox: Exact output grid.

        Returns:
            The collection on the supplied grid, as `io.load_dataset` reads it.

        Raises:
            ConnectionError: No endpoint serves the collection; the message
                names each endpoint's failure.
            AnchorFetchError: The search matched no items.

        Examples:
            >>> recipe.load_raster(anchor, geobox=grid).gs.variables
            ('B04', 'B08')
        """
        failures = []
        for endpoint in self.endpoints:
            try:
                client = StacClient.open(str(endpoint))
                collection = client.collection(self.collection)
            except (APIError, CollectionNotFoundError) as error:
                failures.append(f"{endpoint}: {error}")
                continue
            query = self.query.to_query(self.collection)
            # Recipes fill missing selectors; explicit IDs bypass target filters.
            if query.ids is None:
                if query.bbox is None and query.intersects is None:
                    query = replace(
                        query,
                        bbox=tuple(anchor.geometry.to_crs("EPSG:4326").boundingbox),
                    )
                if query.datetime is None:
                    query = replace(query, datetime=anchor.timespan)
            items = client.search_items(query)
            options = self.load.to_load_kwargs()
            options["patch_url"] = client.patch_url
            return load_dataset(items, geobox=geobox, collection=collection, **options)
        raise ConnectionError(
            f"no endpoint serves {self.collection!r}: " + "; ".join(failures)
        )
