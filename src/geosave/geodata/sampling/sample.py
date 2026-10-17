"""Group Items sharing a footprint, independently of their grids and dates."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import pandas as pd
import shapely

from geosave.geodata.core.vector.sample import GeoSample
from geosave.geodata.core.vector.samples import GeoSamples
from geosave.geodata.stac.translate import item as stac_item
from geosave.geodata.stac.translate import table as stac_table

if TYPE_CHECKING:
    import pystac
    from geosave.geodata import GeoDataFrame

__all__: list[str] = []


def sample(items: gpd.GeoDataFrame) -> GeoDataFrame[GeoSamples]:
    """Build one row per identical normalized WGS84 footprint."""
    if items.crs is None:
        raise ValueError("Item rows have no CRS; set one with set_crs")
    if "collection" not in items or bool(items["collection"].isna().any()):
        raise ValueError("every Item needs a collection")
    rows = cast("gpd.GeoDataFrame", items.to_crs("EPSG:4326").reset_index(drop=True))
    entries = stac_table.to_items(rows)
    groups: dict[bytes, list[int]] = {}
    for position, footprint in enumerate(rows.geometry):
        if footprint is None or footprint.is_empty or not footprint.is_valid:
            raise ValueError("every Item needs a valid nonempty footprint")
        key = footprint.simplify(0).normalize().wkb
        groups.setdefault(key, []).append(position)
    records = []
    for key, positions in groups.items():
        grouped: dict[str, list[pystac.Item]] = {}
        for position in positions:
            entry = entries[position]
            grouped.setdefault(str(entry.collection_id), []).append(entry)
        for group in grouped.values():
            group.sort(
                key=lambda entry: stac_item.read_instant(entry) or pd.Timestamp.min
            )
        records.append(GeoSample.from_items(grouped, footprint=shapely.from_wkb(key)))
    return GeoSamples.from_rows(records, attrs=items.attrs)
