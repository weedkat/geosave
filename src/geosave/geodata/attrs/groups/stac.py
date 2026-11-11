"""What STAC says of a whole raster: its Collection and declared extensions."""

from __future__ import annotations

from typing import Any

import orjson
from pydantic import AliasChoices, Field, field_serializer

from .base import AttrsGroup, Decoded


class StacMetadata(AttrsGroup):
    """Retain the source Collection and declared STAC extensions.

    It lives on a Dataset's root attrs, written by `stac.load_dataset` and read
    by `to_catalog`. Each field is JSON text, which NetCDF, Zarr and GeoTIFF
    tags all hold. Observation properties live on the
    `stac_properties` coordinate instead.

    Attributes:
        collection: The Collection as its STAC JSON, with no links.
        extensions: Schema URIs the Items declare, which the properties need.

    Examples:
        >>> StacMetadata.from_attrs(scene.attrs).collection["id"]
        'sentinel-2-l2a'
    """

    collection: Decoded[dict[str, Any]] | None = Field(
        default=None,
        validation_alias=AliasChoices("stac_collection"),
        serialization_alias="stac_collection",
    )
    extensions: Decoded[list[str]] | None = Field(
        default=None,
        validation_alias=AliasChoices("stac_extensions"),
        serialization_alias="stac_extensions",
    )

    @field_serializer("collection", "extensions")
    def _serialize_as_json(self, value: object) -> str | None:
        """Write a field as JSON text, since NetCDF attrs cannot hold a mapping."""
        return None if value is None else orjson.dumps(value).decode()
