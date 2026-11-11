# CF integration implementation and coverage

This supersedes implementation-status claims in the earlier [capability audit](2026-11-10-cf-xarray-capabilities.md), [API audit](2026-11-10-cf-xarray-api-audit.md) and [field ownership investigation](2026-11-10-cf-field-ownership.md). Those remain evidence of the starting state and dependency behavior.

```python
scene.cf.get_bounds('time')      # native CF relationship
scene.gs.time_coordinate        # CF temporal coordinate, with conventional time fallback
scene.gs.cell_bounds            # one time cell-bounds array, vertex dimension last
scene.gs.time_bounds            # separately recorded observed start/end arrays
scene.gs.support_variables      # structural bounds and CRS carriers
scene.gs.raster_variables       # data variables excluding those structural carriers
scene.gs.variables              # all data-variable names, existing API
```

## Representation contract

CF metadata describes variables; coordinate variables have their own attrs. Attrs groups remain schemas for coherent declarations, not competing storage objects. Bounds and CRS carriers can be native data variables or coordinates. Ancillary observations and cell measures are not automatically excluded as structural metadata. Unknown nonspatial data remains visible to raster validation instead of disappearing through a shape filter.

NetCDF/Zarr retain coordinate names, native bounds names, relationships and support arrays. Observation endpoints remain separate coordinates. Native DataArray extraction cannot carry an extra vertex dimension. GeoSave stacking omits that support and removes dangling coordinate bounds links. Use Dataset selection when full support is required. [CF selection and associated variables](https://cf-xarray.readthedocs.io/en/latest/selecting.html), [bounds](https://cf-xarray.readthedocs.io/en/latest/bounds.html)

## Operation and ownership coverage

| Owner / relevant methods | Implementation or retained responsibility |
|---|---|
| `utils/cf.read_support_variables` | Delegates bounds and CF mapping-name discovery to `.cf`; retains WKT-only GDAL/ODC scalar-carrier extension. Does not use buggy DataArray `.cf.keys()` or assume CRS descriptor construction is validation. |
| `utils/cf.read_time_bounds` | `.cf.get_bounds` resolves the link; GeoSave checks required pair shape and partial NaT. Original storage is unchanged. |
| `utils/cf.check_coordinate_change` | Refuses bounds/cell measures dependent on dimensions whose values would need recomputation. This is an operation guarantee, not a general CF validator. |
| `utils/grid.read_spatial_dims` | CF declarations precede conventional names. Missing declarations may use a conventional axis; ambiguous/conflicting declarations do not silently choose one. Only dimension coordinates establish this affine-axis contract. |
| `utils/grid.write_grid/write_crs` | Native ODC grid placement and CF axis declarations remain explicit. ODC owns affine/GCP placement. Dataset accessor checks support before grid replacement. |
| `GeoDataset._read_variable` | Retains native coordinates; attaches scalar CRS carriers resolved by `.cf`. Does not attach peer ancillary rasters as coordinates during pixel operations. |
| `GeoDataset._select_pixels` | Retains native coordinates and scalar CRS; removes structural data variables from a temporary pixel-operation view. Results restore unaffected support. |
| `GeoDataset.spatial_dims/geobox/validate/_validate_spatial` | Structural data variables no longer invalidate grid discovery. Unknown nonspatial data still fails raster validation. |
| `GeoDataset.mask/_map_variables` | Transform raster values only; preserve support and original coordinate/data-variable placement. Assign changed Variables rather than merging incidental ancillary coordinates. Packing helpers use this path. |
| `GeoDataset.composite` and `transform/temporal.composite` | Native resampler and reduction; preserve independent support without broadcasting it along time. Time-dependent support that needs interpretation is explicitly refused. Observation coverage remains observed, not bin edges. |
| `GeoDataset.reproject/read_pixels` | Resolve mapping carriers even when stored as data variables; preserve independent coordinates/support. Spatial bounds and cell areas requiring recomputation are refused. Singleton thinned windows retain explicit source placement; replacing a CRS removes stale attrs links in favor of ODC encoding. Reprojection preserves the source axis names. |
| `GeoDataset.crop/crop_to_grid` | Native cropping slices dimensions/support; padding/grid replacement has an explicit support guard. No invented bound edges. |
| `GeoDataset.region/write_extent/clip_extent` | Declared extent remains GeoSave/ACDD metadata; structural values are not treated as pixel extents. Native selection does not automatically rewrite declarations. |
| `GeoDataset.to_dataarray/to_numpy/to_tensor/statistics/class_percent` | Exclude structural arrays from bands and summaries; preserve native non-string variable names in pixel traversal. Stacking removes omitted bounds links without mutating the source. |
| `GeoDataset._targets`, RGB and attrs access | Default writes target raster variables; explicit variable targets remain available. Metadata getters can describe all data variables. RGB discovery considers raster variables. |
| `RasterTime.time_coordinate/select_time/timespan`; Dataset `_read_coverage` | Discover temporal coordinates through CF declarations with a conventional-name fallback; preserve native names through selection and STAC stamping. Prefer observed endpoints, then native CF cells for Dataset coverage, then labels. Cell extent is not rewritten as acquisition coverage. Observation endpoints must be scalar or follow the temporal dimension; either endpoint may broadcast independently. Core temporal helpers retain their existing NumPy datetime contract. |
| `GeoDataTree.datasets/validate/geometry/timespan` | Ignore support-only nodes. Resolve bounds referenced by inherited coordinates from parent data variables without mutating the tree. Native groups remain intact. |
| `GeoDataTree.crop/_map_variables/to_dataset` | Operate on raster nodes/variables; support-only parents survive; dimension comparisons exclude structural variables. Xarray merge still checks conflicting values. |
| `io/odc.load_dataset/stamp_band_attrs`, STAC raster/band stamping | Retained native ODC loader, source scale agreement, effective nodata and STAC translation. Time declarations are on coordinate attrs. No `.cf` feature replaces STAC metadata policy. |
| `io/rasterio.open_raster` | Newly reconstructed time coordinates carry CF standard_name and T axis. Native band enumeration remains valid at this format boundary. |
| `io/formats/netcdf` | Native CF decoding and storage; no temporal representation rewrite or automatic spatial rename. |
| `io/formats/zarr` | Same native preservation; written-order restoration retains coordinates with independent dimensions. Fill-encoding and stored-order loops continue to include all native variables. |
| `io/formats/geotiff` | Scalar CRS data variables are promoted for writing, never numbered as image bands. Extra-dimensional CF support is refused; callers choose a format that preserves it or explicitly extract a band. |
| `io/assets.split_items`, `io/paths`, `io/stac` band enumeration/renaming/publication | Structural arrays are retained as support instead of becoming extra assets, bands or naming inputs. |
| Raster plotting / Explorer band discovery | Structural data variables are not offered as display bands; spatial-axis inspection and pixel identification use the raster contract. GeoSave map UI remains separate from `.cf.plot`. |
| `model.processing.to_tensor/validate` | Structural arrays are excluded from implicit tensor bands and raster-dimension checks. Explicit band selections retain their existing behavior. |
| Quantity, Spectral, Packing, GDALBand, ACDD, Extent, StacMetadata | Retained declarations and domain rules; no unconditional conversion to coordinates. Spectral-axis representation is not silently inferred from per-band attrs. |
| Legend | Combined mask/value declarations allow repeated codes; bitfields no longer pretend to be simple value-to-class dictionaries. Native `.cf` expressions handle their interpretation. Palette-only and malformed categorical declarations remain guarded. |
| Native packing transforms | Xarray CF codecs remain responsible for decode/encode. `.cf` offers neither a substitute pixel packing codec nor automatic metadata conflict policy. |

The relevant dependency contracts were rechecked against [axes](https://cf-xarray.readthedocs.io/en/latest/coord_axes.html), [grid mappings](https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html), [flags](https://cf-xarray.readthedocs.io/en/latest/flags.html), [selection](https://cf-xarray.readthedocs.io/en/latest/selecting.html) and [bounds](https://cf-xarray.readthedocs.io/en/latest/bounds.html). The [60-entry API audit](2026-11-10-cf-xarray-api-audit.md) records the broader feature inventory. Geometry conversion, gathered compression, vertical formula evaluation, unit conversion and estimated bounds have no equivalent automatic operation in this raster refactor; introducing them would change behavior rather than remove redundancy.

## Breaking changes and deliberate limits

- Removed `to_cf_bounds`, `from_cf_bounds` and the writer `time_bounds=` switch. These conflated observed coverage with cell extent. Native CF bounds now survive ingestion without changing their meaning.
- NetCDF/Zarr readers preserve source axis names. Call `to_yx` explicitly when canonical names are needed.
- Dataset CF cell bounds are separate from observation coverage. Bounds outside the standalone DataArray dimension set cannot accompany that DataArray.
- Recomputing spatial bounds, cell areas or time cell intervals is not silently inferred. Operations requiring it raise before pixel computation. This is narrower than full support for every CF topology/calendar.
- ACDD/Extent are declarations, not live computed views intercepted across every native xarray operation. `gs.timespan`/`region` retain their documented precedence; users explicitly update declarations when their meaning changes.
- No claim of whole-CF conformance or every cf-xarray numerical feature follows from this integration. Native dependency limitations from the API audit remain relevant.

## Independent criticism and regression evidence

A separate review agent reproduced failures during implementation: unrelated bounds broadcast by resampling, dangling bounds after stacking, dropped scalar mappings, ancillary coordinate merge conflicts, support-only DataTree nodes, and loss of independent coordinates. Each finding was addressed in the corresponding owner and covered by bounded probes or workflow tests.

`tests/geodata/core/raster/test_cf_support.py` exercises lazy mask/stack, independent support during composite, cell-based selection, explicit rejection of stale bounds, unknown nonspatial validation, scalar-mapping GeoTIFF round trip, DataTree inheritance, ancillary masking, independent coordinates during spatial operations and NetCDF/Zarr group bounds round trips. `tests/geodata/test_conventions.py` checks independent observed/CF intervals through both stores. Legend and model-input regressions cover repeated masked codes and exclusion of structural tensor bands.

Final regression evidence:

- Geodata plus model processing/ingestion: **1,265 passed, 24 deselected, 3 xfailed**, 96 warnings. The deselections follow the default slow/integration exclusions; the three existing xfails are chunked warp numerical comparisons.
- After the final observation-endpoint dimension guard: temporal, CF support, STAC raster and DataTree regression selection **105 passed, 2 deselected**, 14 warnings. An initially over-strict guard rejected independently scalar endpoints; the corrected version preserves broadcasting and rejects unrelated axes.
- Explorer raster-layer regression selection after native-axis pixel inspection: **21 passed** with the Bokeh backend initialized. Isolated execution without backend setup fails in existing drawing tests; the new fixture explicitly declares its renamed CF axes.
- Scoped Ruff checks pass. BasedPyright reports **0 errors, 0 warnings** on the changed raster/accessor, CF/grid, warp/temporal, Legend, storage/publication, STAC and model processing modules, with final temporal and Explorer files rechecked separately.
- Local storage checks ran outside the restricted sandbox after sandboxed storage tests stalled. No remote-service or browser integration coverage is claimed.

The broad run preceded the last endpoint guard and Explorer pixel-inspection edit; their focused checks supplement it. This is not an unfiltered whole-repository pass or a claim of full CF conformance.
