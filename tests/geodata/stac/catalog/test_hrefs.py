"""An item table states its assets by absolute href, whatever its file stored."""

import pandas as pd

from geosave.geodata import io, stac
from geosave.geodata.stac.catalog import create_catalog, resolve_hrefs
from tests.geodata.stac.conftest import entry


def test_a_file_stored_with_relative_hrefs_opens_with_absolute_ones(tmp_path):
    from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

    item = entry().to_dict()
    item["assets"]["red"]["href"] = "./scenes/red.tif"
    path = tmp_path / "foreign.parquet"
    to_parquet(parse_stac_items_to_arrow([item]).read_all(), path)

    opened = io.open_vector(path)

    assert opened["assets"][0]["red"]["href"] == str(tmp_path / "scenes" / "red.tif")
    assert stac.StacTable(path).search_items({})[0].get_self_href() is None


def test_a_table_of_absolute_hrefs_is_returned_as_it_is():
    catalog = create_catalog([entry()])

    assert resolve_hrefs(catalog, "s3://bucket/catalog.parquet") is catalog


def test_a_merge_that_drops_the_footer_keeps_asset_hrefs():
    catalog = create_catalog([entry()])
    splits = pd.DataFrame({"id": catalog["id"], "split": "train"})

    merged = catalog.merge(splits, on="id")

    assert merged.attrs == {}
    assert merged.gs.to_items()[0].assets["red"].href == "/tmp/red.tif"
