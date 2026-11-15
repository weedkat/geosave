from typing import TYPE_CHECKING, assert_type

from geosave.geodata.stac.catalog import create_catalog


if TYPE_CHECKING:
    import pystac

    from geosave import GeoCatalog, open_vector
    from geosave.geodata import Dataset
    from geosave.geodata.core import GeoCatalogItem
    from geosave.geodata.io.vector import geoparquet, ogr

    def check_row_reader(row: GeoCatalogItem) -> None:
        assert_type(row.to_item(), pystac.Item)

    def check_accessors(item: pystac.Item) -> None:
        catalog = create_catalog([item])
        assert_type(catalog.gs, GeoCatalog)
        assert_type(catalog.gs.to_items(), list[pystac.Item])
        assert_type(catalog.gs.to_dataset(), Dataset)

        # File readers retain the same accessor for feature and Item rows.
        assert_type(open_vector("items.parquet").gs, GeoCatalog)
        assert_type(geoparquet.open_geoparquet("items.parquet").gs, GeoCatalog)
        ogr.open_geojson("plots.geojson").gs.crs
        ogr.open_geopackage("plots.gpkg").gs.crs


def test_native_containers_dispatch_in_a_fresh_process():
    import subprocess
    import sys
    from textwrap import dedent

    code = dedent("""
        from datetime import UTC, datetime
        import geopandas as gpd
        import numpy as np
        import pystac
        import xarray as xr
        from shapely.geometry import box, mapping
        import geosave as gs
        from geosave.geodata.stac.catalog import create_catalog

        band = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
        dataset = band.to_dataset(name="red")
        tree = xr.DataTree.from_dict({"image": dataset})
        vector = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        item = pystac.Item(id="image", geometry=mapping(box(0, 0, 1, 1)),
                          bbox=[0, 0, 1, 1], datetime=datetime(2024, 1, 1, tzinfo=UTC),
                          properties={}, collection="image")
        item.add_asset("image", pystac.Asset(href="/data/image.tif"))
        catalog = create_catalog([item])
        for value, accessor in [(band, gs.GeoDataArray), (dataset, gs.GeoDataset),
                                (tree, gs.GeoDataTree), (vector, gs.GeoCatalog),
                                (catalog, gs.GeoCatalog),
                                (catalog.iloc[0], gs.GeoCatalogItem)]:
            assert type(value.gs) is accessor
        assert isinstance(catalog, gpd.GeoDataFrame)
        assert type(catalog.iloc[:0].gs) is gs.GeoCatalog
        assert type(vector.iloc[:0].gs) is gs.GeoCatalog
    """)
    subprocess.run([sys.executable, "-c", code], check=True)
