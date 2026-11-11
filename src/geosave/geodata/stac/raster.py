"""Translate STAC source metadata onto native xarray rasters."""

from __future__ import annotations

import json
from collections.abc import Sequence, Hashable
from typing import Any

import numpy as np
import pystac
import xarray as xr

import geosave.geodata.attrs as attrs
from geosave.geodata.conventions import (
    END_DATETIME,
    START_DATETIME,
    STAC_PROPERTIES,
    TIME_COORDINATE,
)
from geosave.geodata.stac.item import (
    read_instant,
    read_item_period,
    select_item_properties,
)


def stamp_properties_coordinate[T: xr.Dataset | xr.DataArray](
    raster: T, properties: Sequence[dict[str, Any]], *, dim: Hashable = TIME_COORDINATE
) -> T:
    """Record each date's Item properties as JSON on the `stac_properties` coordinate.

    Args:
        raster: Raster carrying the selected observation coordinate, or a timeless raster.
        properties: One mapping per selected dimension label, in label order.
        dim: Observation coordinate, default "time".

    Returns:
        Raster with `stac_properties` along the selected coordinate, or scalar
        when that coordinate is scalar.

    Raises:
        ValueError: The mappings do not match the selected labels one to one.

    Examples:
        >>> stamp_properties_coordinate(scene, [{"eo:cloud_cover": 20.0}]).stac_properties.item()
        '{"eo:cloud_cover": 20.0}'
    """
    texts = [json.dumps(dict(entry)) for entry in properties]
    labels = raster.coords.get(dim)
    if labels is None or labels.ndim == 0:
        if len(texts) != 1:
            raise ValueError(
                "a scalar observation coordinate takes one properties mapping"
            )
        return raster.assign_coords({STAC_PROPERTIES: texts[0]})
    if len(texts) != labels.size:
        raise ValueError(
            f"{len(texts)} properties mappings for {labels.size} observation labels"
        )
    return raster.assign_coords({STAC_PROPERTIES: (labels.dims[0], texts)})


def read_properties_coordinate(
    raster: xr.Dataset | xr.DataArray,
) -> list[dict[str, Any]]:
    """Read each date's Item properties from the `stac_properties` coordinate.

    Args:
        raster: Raster, with or without the coordinate.

    Returns:
        One mapping per `time` label, in label order; empty without the
        coordinate.
    """
    if STAC_PROPERTIES not in raster.coords:
        return []
    texts = np.atleast_1d(raster.coords[STAC_PROPERTIES].values)
    return [json.loads(str(text)) for text in texts]


def stamp_item_time(raster: xr.Dataset, item: pystac.Item) -> xr.Dataset:
    """Date a raster opened from an Item.

    Time coordinates already in the file are kept; composite bounds come
    from the Item. A timeless Item adds no dates.
    """
    when = read_instant(item)
    if when is None:
        return raster

    if raster.gs.times is None:
        raster = raster.assign_coords(
            time=xr.DataArray(when, attrs={"standard_name": "time", "axis": "T"})
        )

    if "start_datetime" in item.properties and "end_datetime" in item.properties:
        start, end = read_item_period(item)
        # One Item states one period for every label it holds.
        labels = raster.gs.time_coordinate
        raster = raster.assign_coords(
            {
                START_DATETIME: (
                    labels.dims,
                    np.full(labels.shape, start.to_datetime64()),
                ),
                END_DATETIME: (labels.dims, np.full(labels.shape, end.to_datetime64())),
            }
        )

    return raster


def stamp_item_properties(raster: xr.Dataset, item: pystac.Item) -> xr.Dataset:
    """Attach acquisition properties to each dated observation in a raster.

    Undated rasters are unchanged. Time coordinates must already be assigned.
    """
    coordinate = raster.gs.time_coordinate
    if coordinate is None or read_instant(item) is None:
        return raster
    properties = select_item_properties(item.properties)
    labels = np.atleast_1d(coordinate.values)
    return stamp_properties_coordinate(
        raster, [properties] * labels.size, dim=coordinate.name
    )


def build_stac_metadata(
    items: Sequence[pystac.Item], collection: pystac.Collection | None
) -> attrs.StacMetadata:
    """Read Collection metadata and the schemas declared by Items.

    Args:
        items: Nonempty sequence of source Items.
        collection: Collection to retain without its catalog links, or None.

    Returns:
        Metadata for a Dataset root or a DataTree's collection node.
        Observation properties are stored separately on their coordinate.
    """
    if not items:
        raise ValueError("source metadata needs at least one Item")

    fields = None
    if collection is not None:
        fields = collection.to_dict(include_self_link=False, transform_hrefs=False)
        fields["links"] = []
    extensions = []
    for entry in items:
        extensions.extend(entry.stac_extensions)
    return attrs.StacMetadata(
        collection=fields,
        extensions=list(dict.fromkeys(extensions)),
    )
