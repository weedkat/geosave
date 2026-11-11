# CF structure reference: disk, xarray, `.cf`, `.gs`

Status: reference only. No proposals, no decisions. Every output below was
produced by [`cf_structure_reference.py`](cf_structure_reference.py) with
xarray 2026.4.0, cf-xarray 0.11.3, netCDF4 and Zarr v3, using GeoSave's own
writers (`write_crs`, `write_scale_offset`, `write_nodata`, `write_spectral`,
`write_legend`, `write_acdd`, `write_extent`, `io.formats.netcdf`,
`io.formats.zarr`). CF quotes are from CF 1.12 and its conformance document.

## Vocabulary

| Term | Meaning | Source |
|---|---|---|
| variable | Every named array in a netCDF/Zarr store: data, coordinates, bounds, CRS. A file has only variables. | netCDF data model |
| coordinate variable | 1-D variable whose name equals its dimension (`time(time)`, `x(x)`). Must not have `_FillValue` or `missing_value`. | CF §5 |
| auxiliary coordinate variable | Variable named in a data variable's `coordinates` attribute (`stac_properties(time)`). | CF §5 |
| boundary variable | Variable named by a coordinate's `bounds` attribute (`time_bounds(time, bounds)`). Inherits that coordinate's "BI" attributes: `units`, `calendar`, `standard_name`, `axis`, `long_name`, `positive`, … If it states one itself, it must exactly match the parent. | CF §7.1, App. A |
| grid mapping variable | Scalar variable holding CRS parameters, named by a data variable's `grid_mapping` attribute (`spatial_ref`). | CF §5.6 |
| `ds.coords` vs `ds.data_vars` | **xarray's in-memory classification only.** Both are in `ds.variables`. Which one a variable lands in depends on how it was created or decoded. | xarray |
| attrs vs encoding | attrs: metadata xarray shows and keeps. encoding: how a variable is (or will be) stored. Decoding moves consumed CF attributes from attrs into encoding. | xarray |

## The object

```text
<xarray.Dataset>
Dimensions:          (time: 2, y: 2, x: 3, bounds: 2)
Coordinates:
  * time             (time) datetime64[ns] 2025-01-10T10:00 2025-01-20T10:00
    stac_properties  (time) <U51 '{"platform": "sentinel-2a", "eo:cloud_cover": 10.0}' ...
  * y                (y) float64
  * x                (x) float64
    time_bounds      (time, bounds) datetime64[ns] [[01-01, 01-15], [01-15, 02-01]]
    spatial_ref      int32 32633
Data variables:
    red              (time, y, x) uint16 dask
    nir              (time, y, x) uint16 dask
    scl              (time, y, x) uint8 dask
    dem              (y, x) float32 dask
Attributes:
    id, title, geospatial_bounds, geospatial_bounds_crs, stac_collection, stac_extensions
```

`time_bounds` here is a real CF cell (half-month bins); GeoSave writers do not
create it today, so it was assigned by hand with `time.attrs["bounds"]`.

## On disk (raw netCDF4, no xarray decoding)

```text
global: id, title, geospatial_bounds, geospatial_bounds_crs, stac_collection,
        stac_extensions, coordinates="time_bounds"
time(time)          int64  standard_name=time axis=T bounds="time_bounds"
                           units="days since 2025-01-10 10:00:00" calendar=proleptic_gregorian
time_bounds(time, bounds) int64  units="days since 2025-01-01 00:00:00"
x(x)                float64 _FillValue=nan units=metre resolution=10.0 crs=<WKT>
                           standard_name=projection_x_coordinate axis=X
spatial_ref()       int32  grid_mapping_name=transverse_mercator crs_wkt=<WKT> spatial_ref=<WKT> + projection params
red(time, y, x)     uint16 _FillValue=0 scale_factor=1e-4 add_offset=0.0 long_name units=1
                           standard_name cell_methods="time: mean" common_name=red center_wavelength=0.665
                           geospatial_bounds geospatial_bounds_crs
                           coordinates="stac_properties" grid_mapping="spatial_ref"
scl(time, y, x)     uint8  flag_values=[0 8 9] flag_meanings="no_data cloud_medium cloud_high"
                           color_map='{"9":"#FFFFFF"}' long_name ... grid_mapping coordinates
dem(y, x)           float32 _FillValue=nan units=m standard_name ... grid_mapping (no "coordinates": no time dim)
stac_properties(time) string (no attributes)
```

Zarr v3 stores the same attributes per array in `zarr.json`. Differences:
`fill_value` is array metadata; `time_bounds` also gets `calendar`; root adds
`zarr_variable_order`.

## Reopened: where each piece lands

| Piece | plain `xr.open_dataset` | GeoSave `open_netcdf` (default `mask_and_scale=False`) | GeoSave `open_netcdf(mask_and_scale=True)` | GeoSave `open_zarr` |
|---|---|---|---|---|
| `time_bounds` | coords (via global `coordinates`) | coords | coords | coords |
| `time` → bounds reference | `time.attrs["bounds"]` | `time.encoding["bounds"]` | `time.encoding["bounds"]` | `time.encoding["bounds"]` |
| `spatial_ref` | **data_vars** | coords | coords | coords |
| `red` → grid mapping reference | `red.attrs["grid_mapping"]` | `red.encoding["grid_mapping"]` | encoding | encoding |
| `red` pixels | float64, decoded | uint16, stored | float64, decoded | uint16, stored |
| `red` scale/offset/fill | **encoding** | attrs | **encoding** | attrs |
| `scl.flag_values` | `ndarray([0, 8, 9])` | `ndarray` | `ndarray` | `list [0, 8, 9]` |
| `stac_properties` | coords (via `red.coordinates`) | coords | coords | coords |
| root attrs | unchanged | unchanged | unchanged | unchanged |

## Each piece through `.cf` and `.gs`

| Piece | CF on disk | xarray in memory | `.cf` | `.gs` |
|---|---|---|---|---|
| time labels | coordinate variable `time`, `standard_name`, `axis` | index coord `time`, attrs | `.cf.axes["T"]`, `.cf.coordinates["time"]`, `.cf["T"]` | `.gs.time_coordinate`, `.gs.times` |
| time cells | boundary variable `time_bounds(time, bounds)`, `time:bounds` | coord (or data var); reference in attrs **or** encoding | `.cf.bounds` → `{"time": ["time_bounds"], "T": [...]}`, `.cf.get_bounds("time")` (reads either location) | `Dataset.gs.cell_bounds`; `.gs.timespan` falls back to it → `(2025-01-01, 2025-02-01)` |
| observation periods (current GeoSave) | aux coords `start_datetime(time)`, `end_datetime(time)` | coords | not recognized | `.gs.time_bounds`, `.gs.write_time_bounds`; `.gs.timespan` prefers these over CF cells |
| spatial axes | coordinate variables with `axis`, `standard_name`, `units` | index coords | `.cf.axes["X"/"Y"]`, `.cf.standard_names` | `.gs.spatial_dims` → `("y", "x")` |
| CRS | grid mapping variable `spatial_ref`, `red:grid_mapping` | scalar coord; reference in attrs **or** encoding | `.cf.grid_mapping_names` → `{"transverse_mercator": ["spatial_ref"]}`, `.cf.grid_mappings[0].crs` | `.gs.crs`, `.gs.crs_name` → `EPSG:32633`, `.gs.grid_mapping_coords`, `.gs.support_variables` |
| description | `long_name`, `standard_name`, `units` on the variable (CF §3) | attrs | `.cf.standard_names`, `.cf["surface_bidirectional_reflectance"]` | `attrs.Quantity` (standard_name, units, cell_methods); no accessor property |
| cell methods | `cell_methods` on the variable (CF §7.3) | attrs | none public | inside `attrs.Quantity`; composites write it directly |
| packed data / missing data | `scale_factor`, `add_offset` (CF §8.1), `_FillValue`, `missing_value` (CF §2.5.1) | attrs while stored; **encoding** after decoding | none (xarray decodes natively) | `.gs.packing`, `.gs.nodata`: read **attrs only** |
| flags | `flag_values` / `flag_masks` / `flag_meanings` (CF §3.5) | attrs | `scl.cf.flags` (lazy bool Dataset per meaning), `scl.cf == "cloud_high"`, `.cf.isin`, `.cf.is_flag_variable` | `.gs.legend` (flags + `color_map`), `.gs.is_categorical` |
| palette | `color_map` JSON text (not CF) | attrs | none | `Legend.color_map`, `Legend.colors` |
| spectral band | `common_name`, `center_wavelength`, `full_width_half_max` (STAC EO names, not CF) | attrs | none | `.gs.spectral` |
| declared extent | `geospatial_bounds`, `geospatial_bounds_crs` (ACDD), on root **and every variable** | attrs | none | `attrs.Extent`, `.gs.region` |
| discovery | `id`, `title`, … (ACDD) root | root attrs | none | `.gs.acdd` |
| STAC Collection | `stac_collection`, `stac_extensions` JSON text, root | root attrs | none | `attrs.StacMetadata` |
| STAC observation properties | aux coord `stac_properties(time)`, string | coord | none (not CF-recognized) | `.gs.stac_common_properties` (intersection); `stac.raster.read_properties_coordinate` |

## Selection and container views

```text
scene["red"]          coords: spatial_ref, stac_properties, time, x, y
                      time.attrs still says bounds="time_bounds"   (variable absent)
scene[["red"]]        red, spatial_ref, stac_properties, time, x, y          (no time_bounds)
scene.cf[["red"]]     red, spatial_ref, time, time_bounds, x, y              (no stac_properties)
scene.isel(time=0)    time_bounds dims ("bounds",)   — CF scalar-coordinate bounds
```

DataTree with `time`, `time_bounds`, `stac_properties` and root attrs on `/`, pixels on `/r10m`:

```text
node.to_dataset()                     spatial_ref, time, x, y         (indexed coords only)
node.to_dataset(inherit="all_coords") + stac_properties, time_bounds
tree.gs.datasets["r10m"]              + stac_properties, time_bounds
node.attrs                            {}   (attrs are never inherited)
```

## CF conformance observations in what GeoSave writes

Facts from the files above, not proposals.

1. `time_bounds:units` = `days since 2025-01-01 00:00:00` differs from
   `time:units` = `days since 2025-01-10 10:00:00`. CF §7.1: a BI attribute on
   a boundary variable "must exactly match" the parent's. xarray infers each
   variable's time encoding independently.
2. `x:_FillValue = nan` and `y:_FillValue = nan`. CF §5: "A coordinate variable
   must not have the _FillValue or missing_value attributes." xarray adds NaN
   fill to float variables by default.
3. A global `coordinates = "time_bounds"` attribute is xarray's way to
   round-trip a coordinate not attached to any data variable; it is not a CF
   attribute.
4. `x` and `y` carry a full `crs` WKT attribute in addition to the grid
   mapping variable.

## Behaviour of xarray operations (from the earlier probes)

| Operation | `time_bounds` in data_vars | `time_bounds` in coords |
|---|---|---|
| `resample(time="1MS").mean()` | datetimes averaged: wrong values, no error | dropped; `time.attrs["bounds"]` left dangling |
| `groupby("time.month").max()` | each column maxed: wrong values | dropped |
| `mean("time")` | `(bounds,)` left with no time | dropped |
| `isel`, `sel`, `sortby`, `concat` | kept, paired | kept, paired |
| `ds["red"]` | dropped, reference dangles | dropped, reference dangles |
| `to_netcdf` with dangling reference | writes without error | writes without error |

`cf.add_bounds("time")` is Dataset-only. It **estimates** cells from label
midpoints (labels Jan 10, Jan 20, Feb 15 give `[01-05, 01-15]`,
`[01-15, 02-02]`, `[02-02, 02-28]`, with the outer edges extrapolated), names
the vertex dimension `bounds`, stores the result in coords, and sets
`time.attrs["bounds"]`. It does not record acquisition periods or resample bins.
