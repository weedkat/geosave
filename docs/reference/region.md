# Spatial and temporal targets

A `GeoRegion` describes a CRS-aware spatial `extent` and an optional `timespan`.
Its extent is a native ODC `Geometry`, preserving points, polygon boundaries and
holes. It contains no pixels and does not define a grid.
See [Raster grids](../concepts/raster-grids.md) before choosing a target for reprojection.

```python
region = GeoRegion.from_geometry(city, timespan="2025-06")
raster = raster.gs.write_extent(region.extent)
raster.gs.region.extent       # Declared city geometry
raster.gs.geobox.extent       # Pixel-grid footprint
raster.red.gs.region.extent   # Also retained by a selected band
```

A raster carries two boundaries: its grid (`gs.geobox`, the pixel edges) and,
optionally, a declared extent naming its area of interest, such as a city
inside the grid. Loading with `region=` declares the region this way and loads
the whole grid around it; nothing is masked. Publication writes the region as
the Item `geometry` and the grid as each Asset's `proj:shape` and
`proj:transform`; a catalog's `to_datatree()` reads the geometry back as the
region. Clear it with `write_extent(None)` to publish the whole grid.

`write_extent()` changes metadata only. Native spatial slicing, cropping and
reprojection preserve the declared place; call `write_extent()` again when that
place changes, or `write_extent(None)` to fall back to grid footprints. The
region's timespan is always derived from current temporal coordinates.

Dataset variables retain their own extent attrs, and tree nodes remain locatable
when detached. Without root extent metadata, Dataset and DataTree regions union
their spatial members' extents; no common grid is required. Missing metadata
falls back to pixel-grid footprints, never to a computed valid-pixel mask.

Use `GeoRegion.extent` for spatial geometry; the `from_geometry()` constructor
names the input representation.

::: geosave.geodata.core.region.GeoRegion

## Location descriptions

`region.locate()` returns a Place when reverse geocoding finds an address.
This lookup needs network access and does not change the region's extent.

::: geosave.geodata.utils.geocode.Place
