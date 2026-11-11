# CF field ownership and representation

Status: investigation and proposed sequence; no production changes. This follows the [API audit](2026-11-10-cf-xarray-api-audit.md).

```python
ds.coords['time']        # values locating observations
ds.time.attrs           # declarations describing those values
ds.red.attrs            # declarations describing measured pixels
ds.attrs                # declarations describing the dataset
```

AttrsGroup describes a coherent set of attributes. It is not an alternative to coordinate variables: coordinate variables themselves have attributes. No current group warrants unconditional conversion into coordinates.

## Field ownership map

| Group or fields | Current producers / consumers | Intended owner and decision |
|---|---|---|
| Quantity: standard_name, units, cell_methods | `attrs/groups/quantity.py`; STAC EO/raster translation; concrete accessor composite methods | Attributes on the described data variable or coordinate. Preserve group. Axis declarations belong on coordinates, not root attrs. Review cell_methods separately from coordinate identification. |
| Legend: flag_values, flag_masks, flag_meanings, color_map | `stac/extensions/classification.py`; `io/rasterio.py:read_legend`; DataArray write_legend, colorization | Categorical variable attrs. Preserve declaration and palette ownership; delegate flag interpretation where compatible. Combined-mask validation defects are independent of storage location. |
| Packing: scale_factor, add_offset, _FillValue, missing_value | STAC raster translation; ODC ingestion; packing transforms and format codecs | Variable attrs while packed, native encoding after decoding. No coordinate migration. Check physical-value/storage-state transitions rather than blindly moving keys. |
| Spectral: common_name, center_wavelength, full_width_half_max | `stac/extensions/eo.py:read_attrs`; DataArray write_spectral; `io/stac.py` publication | Per-band variable attrs in current Dataset representation. A wavelength(band) coordinate is a possible spectral-cube representation, not a mandatory migration. Common names and response widths do not automatically become CF axis declarations. |
| GDALBand: variable_name, colorinterp | `io/rasterio.py`; `io/formats/geotiff.py` | Format interoperability metadata on a band. Native xarray variable identity remains its name. No coordinate migration. |
| ACDD: identity, title, summary, keywords, institution, license, time_coverage_start/end | Collection ingestion; Dataset write_acdd; GeoTIFF temporal serialization | Root discovery attributes. Coverage summaries require an explicit maintenance policy after operations; they do not replace sample coordinates or bounds. |
| Extent: geospatial_bounds, geospatial_bounds_crs | DataArray/Dataset write_extent and crop; `attrs/groups/extent.py` | Declared extent metadata on the described object. Distinct from pixel placement and grid mapping. Review root/variable propagation, not conversion to coordinate arrays. |
| StacMetadata: collection, common_properties, extensions | `stac/raster.py`; `io/odc.py`; catalog publication | Dataset root attrs for shared information. Per-observation `stac_properties` already uses a scalar or time-aligned coordinate; this is a GeoSave provenance convention, not automatic CF interpretation. |
| time values and standard_name/axis | `io/odc.py`; `stac/raster.py:stamp_item_time`; rasterio time reconstruction | Values in coordinate; declarations in that coordinate's attrs. Audit existing unannotated file coordinates and format reconstruction before any automatic repair. |
| CRS and grid_mapping | ODC grids, format readers/writers, accessor grid discovery | CRS declarations on a mapping variable; relationship on the raster variable (attrs or encoding according to native decoding). Preserve ODC ownership of pixel placement. |
| start_datetime / end_datetime | `RasterTime.write_time_bounds`; ODC Item grouping; STAC stamping; composite; time selection | Already scalar or dimension-aligned coordinates. Resolve observation-coverage semantics before choosing any CF bounds replacement. |
| CF time bounds and bounds reference | `conventions.to_cf_bounds/from_cf_bounds`; NetCDF/Zarr readers/writers | Bounds are support variables linked from the time coordinate. They need not be xarray coordinates; neither location makes them raster bands. |

## Concrete lifecycle issue: temporal intervals

Current flow:

```text
STAC observation intervals / instantaneous observations
    -> start_datetime, end_datetime coordinates
    -> composite: earliest start and latest end of observed inputs
    -> to_cf_bounds: time_bnds plus time.attrs['bounds']
    -> NetCDF / Zarr
    -> from_cf_bounds: split coordinates again
```

`transform/temporal.py:composite` uses native resampling for labels and pixels, then records observed coverage. These intervals are not necessarily resampling bin edges. `conventions.py` currently treats CF bounds as a file-only spelling of those intervals. This is an explicit representation conversion already present, not evidence that attrs groups store coordinate values incorrectly.

Before changing it, distinguish:

- The labeled cell interval represented by a composite.
- Earliest/latest contributing observations, which do not describe gaps or continuous coverage.
- STAC acquisition intervals.
- Whole-dataset coverage summaries.

A January 1 bin label with observations January 3–20 demonstrates the distinction. This observation alone is not a complete CF conformance verdict. Do not infer continuous coverage, synthesize bin edges from observed endpoints, or promise reversible conversion between distinct meanings.

## Smoke evidence

Executed a tiny one-date, 2×2 Dataset using installed xarray/cf-xarray and current GeoSave:

- `write_time_bounds` stores January 3–20 for a January 1 label.
- `to_cf_bounds(...).cf.get_bounds('time')` resolves shape `(time, bnds)`.
- Extracting `dataset.red` drops the extra-dimensional bounds coordinate.
- With bounds as a coordinate, `dataset.gs.spatial_dims` returns `(y, x)`.
- Moving that same bounds variable to data_vars makes spatial_dims return None.
- The existing reverse conversion restores both observed endpoints.

These are small in-memory probes, not persistence round-trip or complete validation tests. No raster files or production source were modified.

## Proposed sequence

1. Record the temporal meanings above before approving any bounds conversion. Agree on Dataset membership: raster variables versus CF support variables. Trace validation, common grid discovery, transforms, tensor conversion and writers. Do not classify every non-spatial variable as disposable metadata or assume every support variable is listed by one `.cf` property.
2. Set temporal meanings and DataArray extraction guarantees. Compare native Dataset bounds with retained per-observation extension coordinates using caller-visible examples. Keep acquisition coverage distinct where meanings differ.
3. Trace ingestion declarations through ODC, STAC and each format reader; distinguish missing declarations from conflicting declarations. Preserve source metadata unless an explicit operation authorizes normalization.
4. Migrate one proven interpretation seam at a time, retaining attrs groups. Verify values, laziness, metadata and applicable persistence round trips for each change.
5. Audit operation lifecycle: slicing, composite, masking, reprojection, variable extraction and publication. Decide which summaries are recomputed, preserved or invalidated, without implying interception of every native xarray operation.

Independent reviewer agrees that wholesale group conversion would be a category error and highlighted support-variable handling, DataArray bounds loss and summary lifecycle as the structural questions. Representation changes remain proposals, not implementation decisions.

## Independent review follow-up

The reviewer also executed an actual monthly composite of January 10 and January 20 observations: output label January 1, exported bounds January 10–20, and bounds absent on extracted `red`. This corroborates the manually constructed interval probe. Reverse conversion also drops the original bounds variable, name, relationship and its metadata; importing external cell bounds must not silently redefine them as observed coverage. The next investigation must include a representative pixel operation and native NetCDF round trip with support variables, before settling a representation.
