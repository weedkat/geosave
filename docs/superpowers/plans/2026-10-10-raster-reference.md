# Raster reference implementation plan

Implement the constructor API approved in this conversation:
`array(pixels, reference=None, *, dims, coords=None, nodata=None)` and
`raster(variables, reference=None, *, coords=None, nodata=None)`.

`Reference` accepts GeoBox, GCPGeoBox, DataArray and Dataset. Native xarray
references resolve their grid through `.gs.geobox`. The constructor supplies
spatial placement and CRS, retaining exact native x/y coordinate values to
avoid rounding differences when aligning sliced rasters. Other coordinates
are supplied explicitly through `coords`; band semantics and provenance are
not inherited.

`raster` follows the existing construction flow: resolve the grid, construct
an xarray Dataset, validate spatial dimensions and shape, apply grid metadata,
and write optional nodata. `array` delegates to it. Use ODC to write grid
mapping encodings and GeoSave's existing attrs header to describe the axes.

Work in the existing checkout because these changes depend on uncommitted
Alpha API work. Preserve unrelated edits and do not commit them.

- [x] Cover native reference types, lazy pixels, sliced spatial coordinates,
  explicit extra coordinates, shape errors, GCP and custom CRS references.
- [x] Export `array` through geodata and the lazy package root.
- [x] Keep vector rasterization's exact target-coordinate restoration.
- [x] Verify GeoTIFF and NetCDF round trips with explicit time coordinates.
- [x] Smoke-test grid adaptation before settling the simplified flow.
