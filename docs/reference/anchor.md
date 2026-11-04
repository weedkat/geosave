# Spatial and temporal targets

A `GeoAnchor` describes CRS-aware geometry and an optional time range. It contains no pixels.
See [Raster grids](../concepts/raster-grids.md) before choosing a target for reprojection.

::: geosave.geodata.core.anchor.GeoAnchor

## Location descriptions

`anchor.locate()` returns a Place when reverse geocoding finds an address.
This lookup needs network access and does not change the anchor's geometry.

::: geosave.geodata.utils.geocode.Place
