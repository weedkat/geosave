"""Keep every asset href of an item table absolute."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .layout import read_assets

if TYPE_CHECKING:
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
    import pyarrow.compute as pc
    from pystac.utils import make_absolute_href

    assets = read_assets(rows)
    if assets is None:
        return rows

    resolved_assets = []
    any_relative = False
    for position in range(assets.type.num_fields):
        asset = assets.field(position)
        href_position = asset.type.get_field_index("href")
        if href_position == -1:
            resolved_assets.append(asset)
            continue
        hrefs = asset.field(href_position)
        absolute = pc.match_substring_regex(hrefs, ABSOLUTE_HREF)
        if pc.all(absolute).as_py() is not False:
            resolved_assets.append(asset)
            continue

        any_relative = True
        resolved_hrefs = []
        for href in hrefs.to_pylist():
            if href is not None:
                href = make_absolute_href(href, location, start_is_dir=False)
            resolved_hrefs.append(href)
        asset_fields = []
        for field_position in range(asset.type.num_fields):
            asset_fields.append(asset.field(field_position))
        asset_fields[href_position] = pa.array(resolved_hrefs, type=hrefs.type)
        resolved_asset = pa.StructArray.from_arrays(
            asset_fields, fields=list(asset.type), mask=asset.is_null()
        )
        resolved_assets.append(resolved_asset)

    if not any_relative:
        return rows
    resolved = pa.StructArray.from_arrays(
        resolved_assets, fields=list(assets.type), mask=assets.is_null()
    )
    rows = rows.copy()
    rows["assets"] = pd.array(resolved, dtype=pd.ArrowDtype(resolved.type))
    return rows
