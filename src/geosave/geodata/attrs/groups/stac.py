"""What STAC says of a whole raster: its Collection and declared extensions."""

from __future__ import annotations

from typing import Any

import orjson
from pydantic import AliasChoices, Field, field_serializer

from .base import AttrsGroup, Decoded


class StacCollection(AttrsGroup):
    """Retain the source Collection and declared STAC extensions.

    It lives on a Dataset's root attrs, written by `stac.load_dataset`. Each field is JSON text, which NetCDF, Zarr and GeoTIFF
    tags all hold. Item properties are coordinates marked `stac_property`.

    Attributes:
        collection: The Collection as its STAC JSON, with no links.
        extensions: Schema URIs the Items declare, which the properties need.

    Examples:
        >>> StacCollection.from_attrs(scene.attrs).collection["id"]
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


class StacProperty(AttrsGroup):
    """Mark a coordinate as one STAC Item property.

    It lives on the coordinate's own attrs, so the mark follows the coordinate
    through selection, concatenation and files.

    Attributes:
        name: The STAC property name, such as "eo:cloud_cover".

    Examples:
        >>> StacProperty(name="platform").to_attrs()
        {'stac_property': 'platform'}
    """

    name: str = Field(
        validation_alias=AliasChoices("stac_property"),
        serialization_alias="stac_property",
    )
