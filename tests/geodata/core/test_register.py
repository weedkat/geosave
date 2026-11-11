from typing import TYPE_CHECKING, assert_type

from geosave.geodata.stac.catalog import create_catalog


if TYPE_CHECKING:
    import pystac

    from geosave import GeoCatalog, open_vector
    from geosave.geodata import DataTree
    from geosave.geodata.core import GeoCatalogItem
    from geosave.geodata.io.formats import geojson
    from geosave.geodata.io.formats import geopackage
    from geosave.geodata.io.formats import geoparquet

    def check_row_reader(item: GeoCatalogItem) -> None:
        assert_type(item.to_datatree(), DataTree)

    def check_accessors(item: pystac.Item) -> None:
        catalog = create_catalog([item])
        assert_type(catalog.gs, GeoCatalog)
        assert_type(catalog.gs.to_datatree(), DataTree)

        # File readers retain the same accessor for feature and Item rows.
        assert_type(open_vector("items.parquet").gs, GeoCatalog)
        assert_type(geoparquet.open_geoparquet("items.parquet").gs, GeoCatalog)
        open_vector("items.parquet").gs.sample()
        geoparquet.open_geoparquet("items.parquet").gs.sample()
        geojson.open_geojson("plots.geojson").gs.crs
        geopackage.open_geopackage("plots.gpkg").gs.crs


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
        samples = catalog.gs.sample()
        assert samples[0].gs.to_items()[0].id == item.id
        for value, accessor in [(band, gs.GeoDataArray), (dataset, gs.GeoDataset),
                                (tree, gs.GeoDataTree), (vector, gs.GeoCatalog),
                                (catalog, gs.GeoCatalog),
                                (catalog.iloc[0], gs.GeoCatalogItem)]:
            assert type(value.gs) is accessor
        assert isinstance(catalog, gpd.GeoDataFrame)
        assert isinstance(samples[0], gpd.GeoDataFrame)
        assert type(catalog.iloc[:0].gs) is gs.GeoCatalog
        assert type(vector.iloc[:0].gs) is gs.GeoCatalog
    """)
    subprocess.run([sys.executable, "-c", code], check=True)
