"""What STAC says of a whole raster: its Collection and what all its Items share."""

from __future__ import annotations

from typing import Any

import orjson
from pydantic import AliasChoices, Field, field_serializer

from .base import AttrsGroup, Decoded


class StacMetadata(AttrsGroup):
    """Name the STAC Collection a raster belongs to and what all its Items state.

    It lives on a Dataset's root attrs, written by `io.load_dataset` and read
    by `to_catalog`. Each field is JSON text, which NetCDF, Zarr and GeoTIFF
    tags all hold. Properties that differ between dates live on the
    `stac_properties` coordinate instead.

    Attributes:
        collection: The Collection as its STAC JSON, with no links.
        common_properties: Item properties every Item states with one value.
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
    common_properties: Decoded[dict[str, Any]] | None = Field(
        default=None,
        validation_alias=AliasChoices("stac_common_properties"),
        serialization_alias="stac_common_properties",
    )
    extensions: Decoded[list[str]] | None = Field(
        default=None,
        validation_alias=AliasChoices("stac_extensions"),
        serialization_alias="stac_extensions",
    )

    @field_serializer("collection", "common_properties", "extensions")
    def _serialize_as_json(self, value: object) -> str | None:
        """Write a field as JSON text, since NetCDF attrs cannot hold a mapping."""
        return None if value is None else orjson.dumps(value).decode()
