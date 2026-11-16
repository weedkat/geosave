"""How an item table is laid out: its columns, its footer and its Arrow form."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd

if TYPE_CHECKING:
    import pyarrow as pa

# The columns every STAC Item row has, as stac-geoparquet names them.
ITEM_COLUMNS = ("id", "links", "assets", "stac_extensions")
# Key of the Collections, in `rows.attrs` and in the Parquet file metadata.
FOOTER_KEY = "stac-geoparquet"


def read_assets(rows: gpd.GeoDataFrame) -> pa.StructArray | None:
    """Return the assets column as one Arrow struct of asset structs.

    Returns:
        One field per asset key, or None for a table without rows.
    """
    import pyarrow as pa

    assets = pa.Table.from_pandas(rows[["assets"]], preserve_index=False)["assets"]
    if not pa.types.is_struct(assets.type):
        return None
    return cast("pa.StructArray", assets.combine_chunks())


def to_arrow(rows: gpd.GeoDataFrame) -> pa.Table:
    """Return an item table as the Arrow table stac-geoparquet reads and writes.

    Returns:
        The rows in WGS84 with WKB geometry, plus the `bbox` struct column
        the frame does not keep: its geometry states it.
    """
    import pyarrow as pa

    wgs84_rows = rows.to_crs("EPSG:4326")
    if wgs84_rows.active_geometry_name != "geometry":
        wgs84_rows = wgs84_rows.rename_geometry("geometry")
    table = pa.table(wgs84_rows.to_arrow(index=False))

    # GeoPandas names bounds minx..maxy; the struct uses xmin..ymax.
    bounds = wgs84_rows.geometry.bounds
    bbox = pa.StructArray.from_arrays(
        [
            pa.array(bounds["minx"]),
            pa.array(bounds["miny"]),
            pa.array(bounds["maxx"]),
            pa.array(bounds["maxy"]),
        ],
        names=["xmin", "ymin", "xmax", "ymax"],
    )
    return table.append_column("bbox", bbox)
