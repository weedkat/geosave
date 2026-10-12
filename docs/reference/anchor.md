# Spatial and temporal targets

A `GeoAnchor` describes a grid and time range. It contains no pixels.
See [Raster grids](../concepts/raster-grids.md) before choosing a target for reprojection.

::: geosave.geodata.core.anchor.GeoAnchor

## Location descriptions

`anchor.locate()` returns a Place when reverse geocoding finds an address.
This lookup needs network access and does not affect the anchor's grid.

::: geosave.geodata.utils.geocode.Place
