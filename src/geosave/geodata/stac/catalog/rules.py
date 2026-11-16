"""Rules a STAC Item table can be asked to meet: `rows.gs.validate(stac_items)`."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from geosave.geodata.errors import VectorValidationError

from .collections import read_collections
from .hrefs import ABSOLUTE_HREF
from .layout import ITEM_COLUMNS, read_assets


def stac_items(rows: gpd.GeoDataFrame) -> None:
    """Require STAC Item rows, as stac-geoparquet states them.

    Every Item column is present, each asset has an absolute href, and an id
    is unique within its collection.

    Raises:
        VectorValidationError: A column is missing, an asset has no absolute
            href, or two Items of one collection share an id.
    """
    import pyarrow.compute as pc

    missing = set(ITEM_COLUMNS) - set(rows.columns)
    if missing:
        raise VectorValidationError(
            f"STAC Items need columns {sorted(missing)}; item tables come from "
            "create_catalog, StacClient.search, or reading a stac-geoparquet file"
        )

    assets = read_assets(rows)
    if assets is not None:
        # The assets column holds every key any row has; a row lacking one holds null.
        for position, field in enumerate(assets.type):
            # `struct_field` keeps a row without assets null, unlike `assets.field`.
            asset = pc.struct_field(assets, position)
            unlocated = asset.is_valid()
            if asset.type.get_field_index("href") != -1:
                hrefs = pc.struct_field(asset, "href")
                absolute = pc.match_substring_regex(hrefs, ABSOLUTE_HREF)
                unlocated = pc.and_(unlocated, pc.invert(pc.fill_null(absolute, False)))
            row = pc.index(unlocated, True).as_py()
            if row != -1:
                raise VectorValidationError(
                    f"Row at position {row}: asset {field.name!r} has no absolute href"
                )

    collection = rows["collection"] if "collection" in rows else None
    identities = pd.DataFrame({"collection": collection, "id": rows["id"]})
    repeated = identities[identities.duplicated(keep=False)]
    if not repeated.empty:
        raise VectorValidationError(
            f"Items share an id within their collection: {sorted(set(repeated['id']))}"
        )


def stac_collections(rows: gpd.GeoDataFrame) -> None:
    """Require a definition, in the table's footer, of every collection a row names.

    Raises:
        VectorValidationError: A row names a collection the footer lacks.
    """
    if "collection" not in rows:
        return
    named = set(rows["collection"].dropna())
    missing = named - set(read_collections(rows.attrs))
    if missing:
        raise VectorValidationError(
            f"Rows name collections {sorted(missing)} without their definition; "
            "pass them to create_catalog(collections=...)"
        )


def one_collection(rows: gpd.GeoDataFrame) -> None:
    """Require every row to name the same collection.

    Raises:
        VectorValidationError: The rows name several collections.
    """
    if "collection" not in rows:
        return
    named = list(rows["collection"].dropna().unique())
    if len(named) > 1:
        raise VectorValidationError(
            f"Rows name the collections {named}; select one with "
            "rows[rows['collection'] == '<id>']"
        )
