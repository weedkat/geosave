from odc.geo.geobox import GeoBox

ORIGIN = (500_000.0, 9_000_000.0)


def create_grid(size: int = 64) -> GeoBox:
    x, y = ORIGIN
    return GeoBox.from_bbox(
        (x, y, x + size * 10, y + size * 10),
        crs="EPSG:32749",
        shape=(size, size),
        tight=True,
    )
