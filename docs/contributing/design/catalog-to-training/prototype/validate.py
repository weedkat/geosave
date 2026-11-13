# ruff: noqa
# Throwaway prototype for docs/contributing/design/catalog-to-training.md; not library code.
"""Prototype: check an Item table against the stac-geoparquet contract, in memory."""

from __future__ import annotations

import pandas as pd
from pyproj import CRS

REQUIRED = ("id", "geometry", "links", "assets", "stac_extensions")
TOP_LEVEL = {
    "type",
    "stac_version",
    "stac_extensions",
    "id",
    "geometry",
    "bbox",
    "links",
    "assets",
    "collection",
}
TIME = ("datetime", "start_datetime", "end_datetime")


def find_problems(rows) -> list[str]:
    """Every way `rows` breaks the stac-geoparquet contract; empty when it holds."""
    problems = []
    missing = [column for column in REQUIRED if column not in rows.columns]
    if missing:
        problems.append(f"missing columns {missing}")
        return problems
    if rows.crs is None or not CRS(rows.crs).equals(
        CRS("OGC:CRS84"), ignore_axis_order=True
    ):
        problems.append(f"geometry CRS is {rows.crs}, not OGC:CRS84")
    keys = ["collection", "id"] if "collection" in rows else ["id"]
    duplicated = rows.duplicated(subset=keys)
    if duplicated.any():
        problems.append(f"duplicate ids {rows.loc[duplicated, 'id'].tolist()[:3]}")
    for column in TIME:
        if column in rows and not isinstance(rows[column].dtype, pd.DatetimeTZDtype):
            dtype = rows[column].dtype
            pyarrow = getattr(dtype, "pyarrow_dtype", None)
            if (
                pyarrow is None
                or str(pyarrow).startswith("timestamp") is False
                or getattr(pyarrow, "tz", None) is None
            ):
                problems.append(f"{column} is {dtype}, not a tz-aware timestamp")
    has_instant = (
        rows["datetime"].notna()
        if "datetime" in rows
        else pd.Series(False, index=rows.index)
    )
    has_period = pd.Series(False, index=rows.index)
    if "start_datetime" in rows and "end_datetime" in rows:
        has_period = rows["start_datetime"].notna() & rows["end_datetime"].notna()
    undated = ~(has_instant | has_period)
    if undated.any():
        problems.append(
            f"{int(undated.sum())} rows state neither datetime nor start/end"
        )
    for position, assets in enumerate(rows["assets"]):
        if not assets:
            problems.append(f"row {position} has no assets")
            continue
        for key, asset in assets.items():
            if asset is not None and not asset.get("href"):
                problems.append(f"row {position} asset {key!r} has no href")
    footer = rows.attrs.get("stac-geoparquet")
    if footer is not None:
        if "version" not in footer:
            problems.append("footer metadata has no version")
        for key, collection in footer.get("collections", {}).items():
            if collection.get("id") != key:
                problems.append(
                    f"footer collection {key!r} states id {collection.get('id')!r}"
                )
        if "collection" in rows:
            unknown = set(rows["collection"].dropna()) - set(
                footer.get("collections", {})
            )
            if unknown:
                problems.append(f"no Collection definition for {sorted(unknown)}")
    return problems
