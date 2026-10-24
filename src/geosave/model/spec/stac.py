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
from geosave.geodata.stac.source import StacSourceConfig

from .base import SpecModel, Text


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
    """Describe how to find and load one raster from STAC.

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
    load: StacSourceConfig = Field(default_factory=StacSourceConfig)

    @field_serializer("endpoints")
    def _serialize_endpoints(self, endpoints: tuple[HttpUrl, ...]) -> tuple[str, ...]:
        return tuple(str(url) for url in endpoints)

    @field_validator("load", mode="before")
    @classmethod
    def _validate_primitive_load(cls, value: Any) -> Any:
        if isinstance(value, StacSourceConfig):
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
    def _serialize_load(self, load: StacSourceConfig) -> dict[str, JsonValue]:
        """Serialize native load settings as portable primitive values."""
        return load.model_dump(mode="json")

    @model_validator(mode="after")
    def _validate_endpoints(self) -> Self:
        endpoints = tuple(map(str, self.endpoints))
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("endpoints must not contain duplicate URLs")
        return self

    def load_raster(self, anchor: GeoAnchor, /) -> xr.Dataset:
        """Load a lazy raster on an anchor from the first endpoint serving it.

        An endpoint that cannot be reached, or does not publish the
        collection, passes to the next; a failure while loading does not.

        Args:
            anchor: Grid and datetime window to load.

        Returns:
            The collection on the anchor's grid, as `StacSource.load` reads it.

        Raises:
            ConnectionError: No endpoint serves the collection; the message
                names each endpoint's failure.
            AnchorFetchError: The search matched no items.

        Examples:
            >>> recipe.load_raster(anchor).gs.variables
            ('B04', 'B08')
        """
        failures = []
        for endpoint in self.endpoints:
            try:
                source = StacClient.open(str(endpoint)).source(self.collection)
            except (APIError, CollectionNotFoundError) as error:
                failures.append(f"{endpoint}: {error}")
                continue
            # `set_query` takes CQL2 text and `-field` sorts; the recipe holds
            # their JSON forms, which StacQuery takes as they are.
            # The client's own signer, such as Planetary Computer's, stays:
            # a callable has no YAML form, so the recipe never states one.
            source.config = self.load.model_copy(
                update={"patch_url": source.config.patch_url}
            )
            source.query = self.query.to_query(self.collection)
            return source.load(anchor)
        raise ConnectionError(
            f"no endpoint serves {self.collection!r}: " + "; ".join(failures)
        )
