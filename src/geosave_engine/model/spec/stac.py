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
import xarray as xr

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.stac.client import StacProvider
from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSourceConfig

from .base import SpecModel, Text


class SortConfig(SpecModel):
    """Declare one STAC sort field and direction."""

    field: Text
    direction: Literal["asc", "desc"]


class QueryConfig(SpecModel):
    """Declare the STAC item search owned by a model raster."""

    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, JsonValue] | None = None
    datetime: str | tuple[str, str] | None = None
    filter: dict[str, JsonValue] | None = None
    max_items: PositiveInt | None = None
    limit: PositiveInt | None = None
    ids: tuple[Text, ...] | None = None
    sortby: tuple[SortConfig, ...] | None = None

    def to_query(self, collection: str) -> StacQuery:
        """Build the native query with its collection identity."""
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

    Args:
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
        """Load a lazy raster on an anchor from the first available endpoint.

        Not settled yet: read rasters from a catalog through `cuts.stacks`
        and `row.gs.to_stack()` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "StacRecipe.load_raster is not settled; read rasters from a catalog "
            "with cuts.stacks and row.gs.to_stack()"
        )
