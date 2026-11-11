# Raster extent metadata

An anchor describes a place and time independently of pixel placement. Rename
`GeoRegion.geometry` to `extent`, retaining the native ODC `Geometry` value and
the `from_geometry` constructor. No compatibility alias is required.

An `Extent` attrs group owns ACDD `geospatial_bounds` (WKT) and
`geospatial_bounds_crs`. It preserves coordinate order according to the stated
CRS; ODC geometries use x/y order. The ACDD default CRS is EPSG:4326, whose
serialized coordinates are latitude/longitude. Generated metadata always states
the CRS explicitly. Support two-dimensional, valid, nonempty geometries.

`raster.gs.write_extent(geometry | None)` returns a shallow copy and records or
clears the group on the raster and its spatial variables. DataTree applies this
to its nodes. Pixel values, grid and laziness remain unchanged. Extracting a
variable or detaching a raster node retains the extent.

`raster.gs.region.extent` prefers declared metadata; absent metadata falls back
to grid footprints. A Dataset without root metadata can union its variables'
extents. A tree without root metadata unions its raster nodes' extents. A stored
extent does not require a common affine grid. Temporal coverage stays derived
from the existing timespan implementation.

Extent means the declared place represented by a raster, not a valid-pixel mask.
Native slicing, crop and reprojection preserve that declaration. Call
`write_extent` explicitly to change it. This avoids silently redefining a city
boundary when selecting pixels. It also means callers publishing a spatial
subset must explicitly update its declared extent when that is their intent.

Native STAC scene loading retains each band's explicit extent, using root file
metadata or Item geometry only for bands without one. STAC publication uses
declared extents, falling back to grid footprints.
ODC loading preserves an explicit geopolygon/intersects request when supplied;
otherwise its existing grid-derived coverage remains the fallback. Time-varying
source footprints and temporal-bounds redesign are outside this change.

Alternatives considered: keeping only GeoBox loses the polygon; coordinate
storage participates in xarray alignment. Attrs follow the existing group
pattern, with intentional duplication on variables to preserve native selection.
