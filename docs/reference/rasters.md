# Rasters and trees

Rasters are native xarray objects. Build them with xarray's own constructors and
place them with ODC's `xr_coords` or `wrap_xr`. Start with [Your first raster](../basics/first-raster.md),
then see [Raster data](../basics/rasters.md). Accessor methods are listed
separately for [Dataset](dataset-accessor.md), [DataArray](dataarray-accessor.md),
and [DataTree](datatree-accessor.md).

```python
array = scene.gs.to_array()
wrapped = array.to_dataset(name="image")  # Native wrapping keeps all dimensions.
bands = array.gs.to_dataset()  # Split the labelled "band" dimension.

scene = stac.load_dataset(rows.gs.to_items(), geobox=grid, chunks={})
tree = xr.DataTree.from_dict({"optical": scene})
```

Raster conversion preserves pixels and does not choose an output grid; align
mixed grids with `tree.gs.reproject` first. Native tree constructors keep
attrs on the Datasets supplied at each explicit path. Catalog Items load through `stac.load_dataset`.

Stacking requires matching CF pixel meaning: units (including whether stated),
packing, unsigned interpretation, flags, validity ranges, calendars and cell
methods. The remaining attrs follow xarray's `drop_conflicts`. The stack has no original storage encoding;
choose output encoding explicitly when saving packed analysis results.
Splitting preserves shared attrs on each band, but cannot recover per-band
metadata lost during stacking. Use native `to_dataset(name=...)` for wrapping.

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

NetCDF stores cells as a CF boundary variable, `time_bounds(time,
bounds)` named by `time`'s `bounds` attr, with `time`'s units and calendar.
Zarr stores the start/end coordinates directly. GeoSave's NetCDF reader
turns bounds back into `start_datetime` and `end_datetime`, so
in memory cells are always those two coordinates; a foreign CF file's bounds,
under any name, open the same way.
A monthly bin label is not necessarily the first contributing observation.
`scene.gs.time_coordinate` discovers that coordinate through CF time/axis
declarations, falling back to the conventional `time` name. A declared `date`
coordinate keeps its name through time selection and STAC stamping.

`scene.gs.variables` lists the data variable names. GeoSave's readers and ODC hold the grid mapping as a coordinate; after a plain
`xr.open_dataset`, make it one with `ds.set_coords("spatial_ref")`. CF
structure GeoSave does not model, such as spatial bounds or cell measures,
stays as ordinary variables: read it with `.cf`, and recompute or drop it
yourself after an operation that changes its dimension.

A DataTree aggregates its datasets' timespans; its nodes inherit their
parents' cells along with `time`.

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
series = xr.concat(
    decoded_rasters, dim="time", join="exact", combine_attrs="drop_conflicts",
)
```

`join="exact"` checks indexes, not CRS, units, class meanings or dtype. Prepare
compatible rasters before combining them. Native concat's default attrs policy
is `override`.

For trees, select the Dataset at a caller-defined node before applying temporal
operations. Use `.to_dataset(inherit="all_coords")` when all inherited
coordinates are needed:

```python
observations = tree["optical"].to_dataset()
monthly = observations.gs.mask_and_scale().gs.composite("MS", "mean")
january = monthly.sel(time="2025-01")
```

## Validation

`validate(*rules)` first checks the object's raster structure, then runs each
additional rule without reading pixels. DataArray and Dataset require a grid
ODC recognizes, with a CRS; DataTree requires raster leaves with locally owned
grids. A `Rule[T]` is a callable taking the native object and returning `None`,
or raising `RasterValidationError` with the offending node or variable.

Rules live in `geosave.geodata.core.raster.rules`, organized by object type.
They validate only. Accessors own transformations and metadata preservation;
STAC publication owns observation and asset membership.

```python
from geosave.geodata.core.raster.rules.dataarray import compatible_resampling, decoded
from geosave.geodata.core.raster.rules.dataset import same_pixel_meaning
from geosave.geodata.core.raster.rules.datatree import compatible_concat, same_geobox

band.gs.validate()                       # a georeferenced grid
band.gs.validate(decoded)                # and no active CF packing
scene.gs.validate(same_pixel_meaning)    # one packing, flags and units across variables
tree.gs.validate(same_geobox)           # every raster node on the identical GeoBox
observations.gs.validate(compatible_concat("time"))  # preflight without combining
band.gs.validate(compatible_resampling("mean"))  # class codes need nearest, mode, first or last
```

| Rule | Checks | Used by |
| --- | --- | --- |
| `rules.common.georeferenced` | an ODC grid with a CRS | DataArray/Dataset `validate` |
| `rules.dataarray.decoded` | no active CF packing | `composite` |
| `rules.dataarray.compatible_resampling(method)` | class codes resampled only by nearest, mode, first or last | `reproject`, `composite` |
| `rules.dataset.same_pixel_meaning` | one packing, flags and units across variables | `to_array` |
| `rules.datatree.raster_leaves` | raster data only on leaves with locally owned grids | DataTree `validate` |
| `rules.datatree.same_grid` | matching CRS, resolution and pixel alignment; extents may differ | explicit validation |
| `rules.datatree.same_geobox` | identical CRS, transform and shape across raster nodes | `merge`, `concat`, consumers requiring shared spatial context |
| `rules.datatree.compatible_concat(dim)` | immediate raster children with matching grids, indexes outside `dim`, variable sets and representations | `concat` |

A rule that needs an operation's argument is built from it:
`compatible_resampling(method)` returns the rule that the band's values allow
`method`, which `reproject` and `composite` pass to `validate`. Argument checks, such as a
mask spanning foreign axes, stay inside their operation. Likewise,
`compatible_concat(dim)` checks whether child Datasets can be concatenated;
`concat` calls that rule before combining them. Different units between distinct
variables are allowed; corresponding variables across children must agree.

Numerical channel conversion (`to_numpy` and `to_tensor`) allows different units
and requires explicit dtype preparation when source dtypes differ. It does not
use the storage-stacking rule `same_pixel_meaning`.

The former rule imports from `raster.base`, `raster.dataarray`, `raster.dataset`
and `raster.datatree` have been removed; import them from `raster.rules` modules.
