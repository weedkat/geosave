from typing import TYPE_CHECKING, Any, assert_type

if TYPE_CHECKING:
    import pystac
    from odc.geo.geobox import GeoBox

    from geosave import GeoCatalog, GeoSamples, open_vector, stac
    from geosave.geodata import Dataset, DataTree
    from geosave.geodata.core import GeoItem, GeoSample
    from geosave.geodata.io.vector import geojson, geopackage, geoparquet

    def check_row_readers(item: GeoItem, sample: GeoSample) -> None:
        assert_type(item.to_xarray(), Dataset)
        assert_type(sample.to_xarray(), DataTree)

    def check_accessors(item: pystac.Item, grid: GeoBox) -> None:
        catalog = stac.create_catalog([item])
        assert_type(catalog.gs, GeoCatalog)
        samples = catalog.gs.sample()
        assert_type(samples.gs, GeoSamples)
        assert_type(samples.gs.chip(32, target=grid).gs, GeoSamples)
        assert_type(
            samples.gs.frame(windows={"s2": {"slots": ["0M"], "reduce": "first"}}).gs,
            GeoSamples,
        )

        # File contents determine the accessor at runtime.
        assert_type(open_vector("items.parquet").gs, Any)
        assert_type(geoparquet.open_vector("items.parquet").gs, Any)
        open_vector("items.parquet").gs.sample()
        geoparquet.open_vector("items.parquet").gs.sample()
        geojson.open_vector("plots.geojson").gs.crs
        geopackage.open_vector("plots.gpkg").gs.crs
