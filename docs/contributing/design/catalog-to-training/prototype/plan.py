# ruff: noqa
# Throwaway prototype for docs/contributing/design/catalog-to-training.md; not library code.
"""Prototype: the two planning verbs pandas lacks, and samples as STAC Items."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
from odc.geo.geobox import GeoBox, GeoboxTiles

from geosave.geodata.stac.extensions import projection


def tile(grid: GeoBox, size: int, *, overlap: int = 0) -> gpd.GeoDataFrame:
    """One row per full tile of `grid`: WGS84 footprint plus the exact grid as proj:*."""
    step = size - overlap
    rows = []
    for y in range(0, grid.shape.y - size + 1, step):
        for x in range(0, grid.shape.x - size + 1, step):
            box = grid[y : y + size, x : x + size]
            fields = projection.build_fields(box)
            rows.append(
                {
                    "chip": f"{y // step}_{x // step}",
                    "proj:code": fields.get("proj:code"),
                    "proj:shape": list(fields["proj:shape"]),
                    "proj:transform": list(fields["proj:transform"]),
                    "geometry": box.extent.to_crs("EPSG:4326").geom,
                }
            )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def read_geobox(row: pd.Series) -> GeoBox:
    """The exact grid a planned row states in proj:*."""
    from affine import Affine

    transform = Affine(*row["proj:transform"][:6])
    return GeoBox(tuple(row["proj:shape"]), transform, row["proj:code"])


def window(
    slots: gpd.GeoDataFrame, *, by: str, size: int, step: int, collections: list[str]
) -> gpd.GeoDataFrame:
    """Consecutive slot rows of each group → one sample row with Item id lists."""
    samples = []
    for key, group in slots.sort_values("datetime").groupby(by, sort=False):
        for first in range(0, len(group) - size + 1, step):
            frame = group.iloc[first : first + size]
            head = frame.iloc[0]
            sample = {
                "id": f"{key}_{frame['datetime'].iloc[0]:%Y%m%d}",
                by: key,
                "proj:code": head["proj:code"],
                "proj:shape": head["proj:shape"],
                "proj:transform": head["proj:transform"],
                "start_datetime": frame["datetime"].min(),
                "end_datetime": frame["datetime"].max(),
                "geometry": head["geometry"],
            }
            for collection in collections:
                ids = []
                for value in frame[collection]:
                    ids.append(value if isinstance(value, str) else "")
                sample[collection] = ids
            samples.append(sample)
    return gpd.GeoDataFrame(samples, crs=slots.crs)


def to_sample_items(
    samples: gpd.GeoDataFrame, collections: list[str], collection_id: str
) -> list[pystac.Item]:
    """Samples as STAC Items: grid in proj:*, window as start/end, sources as properties."""
    from shapely.geometry import mapping

    items = []
    for _, row in samples.iterrows():
        properties = {
            "start_datetime": row["start_datetime"].isoformat(),
            "end_datetime": row["end_datetime"].isoformat(),
            "proj:code": row["proj:code"],
            "proj:shape": list(row["proj:shape"]),
            "proj:transform": list(row["proj:transform"]),
        }
        for collection in collections:
            properties[collection] = list(row[collection])
        item = pystac.Item(
            row["id"],
            mapping(row["geometry"]),
            list(row["geometry"].bounds),
            None,
            properties,
            collection=collection_id,
            stac_extensions=[
                "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
            ],
        )
        items.append(item)
    return items
