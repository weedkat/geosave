# Explorer raster loading

```python
# The map's view is a GeoBox: visible bounds from RangeXY, plot pixels from PlotSize
view = GeoBox.from_bbox((left, bottom, right, top), crs="EPSG:3857", shape=(height, width))

row.gs.read_pixels(view, variables=["red"])   # a catalog row: each file from its overview
ds.gs.read_pixels(view)                       # any Dataset: crop, keep every n-th pixel, compute

items.gs.explore()                            # polygon and raster checkboxes per row
```

## Decisions

The explorer stays on HoloViews' own mechanism: one DynamicMap fed by RangeXY,
each raster layer drawing `gv.Image` or `gv.RGB` for the view. A tile server
(rio-tiler behind a Panel route) was prototyped and rejected: it needed a
server for notebook display, a variable-to-asset mapping, a second source of
nodata and colours, and bypassed geosave's credential handling.

`read_viewport` is deleted. Reading a view's pixels belongs to geodata:

- `GeoCatalogItem.read_pixels(geobox, variables=None) -> Dataset` opens each
  data asset holding a requested variable with
  `open_raster(href, chunks=None, overview_level=...)`. odc-geo's
  `compute_reproject_roi` gives the file's window and shrink factor; the
  overview is the coarsest whose factor does not exceed the shrink. Only that
  window is loaded, then reprojected onto `geobox` and merged.
- `GeoDataset.read_pixels(geobox) -> Dataset` crops with the same window,
  keeps every `read_shrink`-th pixel, computes, and reprojects.
- Both return float32 with nodata as NaN, on `geobox`'s grid; a raster the
  view misses returns NaN without reading.

Dask is not used for view reads: on Sentinel-2, `chunks={}` was 3-6x slower
than cropping a non-Dask lazy array and loading it.

## Catalog rows open through rasterio

A row is one scene: its data assets open with the GDAL reader on their native
grids and merge, then take the Item's date and properties. Rows of one
collection join along `time` with `concat`; the catalog accessor builds the
tree. `combine_rasters` no longer serves STAC reads: a row states its grouping,
so nothing is regrouped from file tags. `stac.open_raster` and
`open_collections` give way to `open_scene(item)`, and the tree is built by
`GeoCatalog.to_xarray`. odc-stac stays in `StacSource`, which loads onto a
chosen grid. Rows whose files differ in grid, or a collection whose rows do,
fail at the explicit merge or concat.

## Layer cache

A raster layer reads at a Web Mercator zoom level `z`, pixel size
`156543.03392804097 / 2**z` metres, the level nearest the view's pixel size, over the view padded by half its size on each side. It reads again only
when the view leaves the cached bounds, the zoom level changes, or the date or
shown bands change. The cached image itself is drawn: Bokeh moves and scales
it, so pans and zooms inside it neither read nor resend pixels. Contrast
changes read nothing.
GDAL's block and VSI caches make revisits cheap.

## Catalog layers

`CatalogLayer` is deleted. `VectorLayer` holds three row lists:
`footprint_rows` (polygons drawn; the table's checkbox column, all by default),
`raster_rows` (rows whose rasters are on the map; a `tickCross` column offered
only when rows carry raster assets) and `picked_rows` (a map click's
highlight). A checked row opens as one `RasterLayer` holding
`row.gs.to_xarray()` (bands, date, legend) and the row itself (pixels).
`GeoSamples.explore` is removed; `explorer.add(samples[0])` maps a sample's rows.

## Out of scope

Rows whose assets differ in grid (Sentinel-2 10/20/60 m) still fail
`to_xarray`; per-grid DataTree groups are a separate change. So are deleting
`combine_rasters` and the multi-path `open_raster`, restricting `to_catalog`
to COG assets, and reading views off the page's thread.
