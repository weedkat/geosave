"""Keep every asset href of an item table absolute."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .layout import read_assets

if TYPE_CHECKING:
    import pyarrow as pa

    from geosave.geodata import GeoDataFrame

# An href naming its own location: a URL scheme or a rooted path.
ABSOLUTE_HREF = r"^([A-Za-z][A-Za-z0-9+.-]*:|/)"


def resolve_hrefs(rows: GeoDataFrame, location: str) -> GeoDataFrame:
    """Make every relative asset href absolute against the file the rows came from.

    An item table states its assets by absolute href. A file another writer
    stored with relative ones is resolved once, as it is read.

    Args:
        rows: Item table as its file stores it.
        location: Absolute path or URL of that file.

    Returns:
        The table with absolute asset hrefs; `rows` itself when none is relative.

    Examples:
        >>> resolve_hrefs(rows, "s3://bucket/s2/catalog.parquet")["assets"][0]["B04"]["href"]
        's3://bucket/s2/20250601T103000/B04.tif'
    """
    import pyarrow as pa

    assets = read_assets(rows)
    if assets is None:
        return rows

    resolved_assets = []
    any_resolved = False
    for position in range(assets.type.num_fields):
        asset = assets.field(position)
        if has_relative_href(asset):
            asset = resolve_asset(asset, location)
            any_resolved = True
        resolved_assets.append(asset)
    if not any_resolved:
        return rows

    resolved = pa.StructArray.from_arrays(
        resolved_assets, fields=list(assets.type), mask=assets.is_null()
    )
    rows = rows.copy()
    rows["assets"] = pd.array(resolved, dtype=pd.ArrowDtype(resolved.type))
    return rows


def has_relative_href(asset: pa.StructArray) -> bool:
    """Return whether any row states this asset by a relative href."""
    import pyarrow.compute as pc

    if asset.type.get_field_index("href") == -1:
        return False
    absolute = pc.match_substring_regex(asset.field("href"), ABSOLUTE_HREF)
    return pc.all(absolute).as_py() is False


def resolve_asset(asset: pa.StructArray, location: str) -> pa.StructArray:
    """Return one asset's column with every href absolute against `location`.

    Args:
        asset: One asset key's struct, a row each, with an `href` field.
        location: Absolute path or URL of the file the rows came from.
    """
    import pyarrow as pa
    from pystac.utils import make_absolute_href

    href_position = asset.type.get_field_index("href")
    hrefs = asset.field(href_position)
    resolved_hrefs = []
    for href in hrefs.to_pylist():
        if href is not None:
            href = make_absolute_href(href, location, start_is_dir=False)
        resolved_hrefs.append(href)

    asset_fields = []
    for position in range(asset.type.num_fields):
        asset_fields.append(asset.field(position))
    asset_fields[href_position] = pa.array(resolved_hrefs, type=hrefs.type)
    return pa.StructArray.from_arrays(
        asset_fields, fields=list(asset.type), mask=asset.is_null()
    )
