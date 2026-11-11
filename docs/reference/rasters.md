# Rasters and trees

Rasters are native xarray objects. Build them with xarray's own constructors and
place them with `write_grid`. Start with [Your first raster](../first-raster.md),
then see [Raster data](../basics/rasters.md). Accessor methods are listed
separately for [Dataset](dataset-accessor.md), [DataArray](dataarray-accessor.md),
and [DataTree](datatree-accessor.md).

```python
array = scene.gs.to_dataarray()
wrapped = array.rename("image").gs.to_dataset()  # Keep the band dimension.
bands = array.gs.to_dataset(dim="band")  # Split bands into variables.

tree = xr.DataTree.from_dict({"/optical": scene})
scene = tree.gs.to_dataset()  # Compatible same-grid nodes only.
```

Raster conversion preserves pixels and does not choose an output grid. For
mixed-grid trees, pass `grid=` explicitly, as `tree.gs.to_dataset(grid="r10m")`. Native tree constructors keep
attrs on the Datasets supplied at each explicit path. Catalog-row
`gs.to_datatree()` remains the reader that opens raster assets.

## Temporal bounds

Dataset and DataArray accessors own observation bounds:

```python
dated = monthly.gs.write_time_bounds("MS")              # labels start their cells
weekly = weekly.gs.write_time_bounds("W", label="right")  # labels end their cells
dated.start_datetime, dated.end_datetime
start, end = dated.gs.timespan
```

```text
time        start_datetime  end_datetime
2025-01-01  2025-01-01      2025-02-01
2025-02-01  2025-02-01      2025-03-01
```

A cell is the `start_datetime` and `end_datetime` coordinates on the time
coordinate's dimensions, the names STAC Items use, on DataArrays and Datasets
alike. Cells are half-open like resample bins: each ends where the next
starts. `write_time_bounds` states one `freq` step per label without reading
pixels, and refuses a label that is not on an edge of `freq`; `dim=` names the
label coordinate when it is not the CF time coordinate. Readers state the
periods their source records, such as each loaded Item's own period.
`timespan` reduces cells, falling back to time labels, and returns `None` for
an empty time axis.

NetCDF and Zarr store cells as a CF boundary variable, `time_bounds(time,
bounds)` named by `time`'s `bounds` attr, with `time`'s units and calendar.
GeoSave's readers turn it back into `start_datetime` and `end_datetime`, so
in memory cells are always those two coordinates; a foreign CF file's bounds,
under any name, open the same way.
A monthly bin label is not necessarily the first contributing observation.
`scene.gs.time_coordinate` discovers that coordinate through CF time/axis
declarations, falling back to the conventional `time` name. A declared `date`
coordinate keeps its name through time selection and STAC stamping.

`scene.gs.raster_variables` excludes structural CF bounds and CRS carriers;
`scene.gs.support_variables` lists those carriers, including coordinates.
`scene.gs.variables` continues to list all data variables. Ancillary rasters
and cell measures remain data variables; unknown nonspatial data is not
silently discarded by raster validation.

Masking preserves structural values. Coordinate-changing operations refuse
support that would need recomputation, rather than keeping stale bounds or
cell areas. Compositing with time-dependent CF bounds currently requires an
explicit prior choice of their treatment. Native DataArray extraction cannot
retain extra-dimensional bounds; `.gs.to_dataarray()` also removes references
to omitted bounds. Keep a Dataset when those relationships are required.
GeoTIFF output refuses extra-dimensional CF support; NetCDF/Zarr preserve it.

A DataTree aggregates its datasets' timespans. Write bounds on individual
Datasets before assigning them to the tree; children may have different time
axes, so there is no tree-wide bounds writer.

## Composites and time selection

Decode stored values explicitly before reducing them:

```python
decoded = observations.gs.mask_and_scale()
monthly = decoded.gs.composite("MS", "mean", properties="intersect")
january = monthly.sel(time="2025-01")
```

`composite` uses native resampling; each output cell is its resample bin, and
scene properties are rebuilt for each bin. Input cells are replaced, not
reduced. `closed`, `label`, `origin` and `offset` follow native resample, and
`label` only picks which bin edge becomes the time label. Empty bins keep their
cell and, without interpolation, NaN pixels. Select with native `.sel` on
labels, or on `start_datetime`/`end_datetime` for overlap. Pixels stay lazy.

Class codes permit only `first` and `last`, without interpolation. Existing `cell_methods` are retained
and the temporal method is appended. Variance removes `units`; state the known
squared units explicitly on its output. Native sum returns zero for an observed
bin whose pixels are all missing. Reduction does not restore the input's integer
storage encoding.

Select the resampled dimension explicitly when it is not named `time`. Optional
interpolation fills interior pixel gaps after reduction:

```python
monthly = decoded.gs.composite(
    "MS", "mean", dim="acquired", interpolation="linear",
)
```

`dim` defaults to `"time"`. Interpolation accepts `"linear"` or `"nearest"`,
uses coordinate distances, preserves finite values and never extrapolates.
It fills pixel gaps, including missing pixels within observed groups, without
inventing observation properties. Lazy output remains lazy, but the
selected dimension is rechunked into one chunk, increasing per-task memory.
Native `.interpolate_na()` remains available for separate interpolation
with explicit gap limits or extrapolation.

Use xarray directly for concatenation:

```python
from geosave.geodata import attrs

series = xr.concat(
    decoded_rasters, dim="time", join="exact", combine_attrs=attrs.combine_attrs,
)
```

`join="exact"` checks indexes, not CRS, units, class meanings or dtype. Prepare
compatible rasters before combining them. The callback follows native
`drop_conflicts` and warns about omitted keys; native concat's default without
it is `override`.

For trees, retrieve each node's Dataset with its inherited coordinates and
apply these methods there. Nodes may have different grids and time axes.
Explicit inheritance includes non-index observation bounds and scene properties:

```python
observations = tree["optical"].to_dataset(inherit="all_coords")
monthly = observations.gs.mask_and_scale().gs.composite("MS", "mean")
january = monthly.sel(time="2025-01")
```

## Validation

Basic validation checks nonempty spatial structure, coordinate consistency and
any placement ODC discovers. Unreferenced pixel arrays are valid; pixels are
never computed or repaired by validation.

```python
image.gs.validate()
image.gs.validate(georeferenced=True)
scene.gs.validate(same_dims=True, same_grid=True)
tree.gs.validate(georeferenced=True, same_grid=True)
```

DataArray accepts only `georeferenced`. Dataset and DataTree additionally accept
`same_dims` and `same_grid`, comparing variables and data-bearing nodes
respectively. Different grids are allowed unless requested otherwise.
`same_grid=True` requires known equal grids; absent grids are not evidence of
shared placement. `same_dims=True` compares spatial axis names.

ODC owns spatial metadata interpretation. GeoSave checks regular coordinate
spacing because ODC's affine inference assumes it, and checks that returned
transforms are finite and nonsingular. GCP placement remains valid; operations
that need a regular GeoBox reject it at their own boundary. There is no
`affine`, `spatial` or `allow_missing_grid` validation flag.

Spatial transforms, TIFF writing, STAC Item construction, model restoration and
map placement invoke this validator before interpreting pixel placement. Each
operation keeps its own additional requirements, such as a CRS for reprojection
or a single date for an Item asset. Path planning only resolves selections and
destinations; writers perform validation before I/O.
