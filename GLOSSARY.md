# GeoSave

GeoSave connects native geospatial data with model development. Catalogs index data; they are optional for processing a raster.

## Language

### Data and models

**Catalog**: A table of geospatial Items and their assets, used to discover and
select data. It is not a model input tensor.

**Item**: One STAC observation, as publication writes it: the positions along
`split_item` (time by default) make separate Items.
_Avoid_: observation row, inventory row

**Asset**: One file within an Item. `split_asset` divides an Item into assets:
`"variable"` gives one per data variable, and a dimension one per label. A
lone raster's sole asset is named `asset`; a tree's asset names start with
the node path.
_Avoid_: inventory, layer

**Sample**: The inputs belonging to one model example, with targets when labelled.
A sample may contain one observation or a temporal window; it need not be a catalog row.

**Scene**: Imagery associated with one observation. This does not imply one file,
one grid, or one model-sized chip. Use explicit dimensions and grids to describe
what a model accepts.

**Grid**: For an affine raster, the CRS, affine transform and array shape that
place its pixels. Resolution alone does not identify a grid. A tree's grid
node names are local labels; `tree.gs.grids` summarizes their actual metadata.
Publication selects Collection boundaries with `collections=["s2", "dem"]`;
neither hierarchy depth nor grid inspection implies Collection membership.

**Recipe**: A ModelSpec YAML declaration of invocation inputs and dependent
processing calls. It uses installed operations; it is not model weights or a
training loop. See [the execution guide](docs/guides/model-spec.md).

### Raster structure

Terms follow CF, xarray and cf-xarray. See
[the CF structure reference](docs/contributing/reviews/2026-11-11-cf-structure-reference.md).

**Variable**:
Any named array of a Dataset: pixels, coordinates, bounds, the CRS. A netCDF or
Zarr store holds only variables; xarray lists all of them in `ds.variables`.

**Coordinate**:
A variable in `ds.coords`. Which variables are coordinates is xarray's
in-memory classification, set by construction or decoding, not a CF property.
A CF *coordinate variable* is the 1-D one named after its dimension (`time`,
`x`); an *auxiliary coordinate* is named by a variable's `coordinates`
attribute (`stac_properties`).

**Data variable**:
A variable in `ds.data_vars`. A **raster variable** is a data variable holding
pixels; bounds and grid mapping variables are support variables, not rasters.
_Avoid_: layer

**Band**:
One spectral or file channel. It is a raster variable in a Dataset, or a
position along the `band` dimension of a stacked DataArray.

**Reference**:
An attribute naming another variable: `bounds`, `grid_mapping`, `coordinates`,
`cell_measures`. After decoding, xarray may keep it in encoding instead of
attrs; `.cf` reads either.

**Attrs**:
Metadata xarray shows and carries with a variable or Dataset.
**Encoding**:
How a variable is stored. Decoding moves consumed CF attributes, such as
`scale_factor` or `bounds`, from attrs into encoding.

**Root attrs**:
Attrs of a Dataset or DataTree node itself. A DataArray taken from a Dataset
and a child node of a tree do not inherit them.

**Attr group**:
A validated, centrally readable set of attrs that mean one thing together
(`Packing`, `Legend`, `Spectral`, `Description`, `ACDD`, …). It lives beside
`.cf` as a structured view of the same attrs, not a separate store. A
`gs.write_*` method states a value as the right attrs, encoding or variables.

**Cell**:
The period one time label stands for, as two coordinates on the time
dimension: `start_datetime(time)` and `end_datetime(time)`. A cell ends where
the next starts (CF convention). They survive `ds[[...]]` and travel with a
DataArray.
_Avoid_: estimating cells from labels

**Bounds**:
The CF boundary variable a netCDF or Zarr file stores cells as,
`time_bounds(time, bounds)`, named by `time`'s `bounds` attribute and sharing
its `units` and `calendar`. GeoSave's readers turn it into a cell's two
coordinates; it does not exist in memory.

**Cell methods**:
The CF `cell_methods` attribute stating how a variable's values were reduced
over cells, such as `"time: mean"`.

**Observation**:
One label along the time dimension. A loader's grouping (`solar_day`, `id`)
decides which source Items it combines.

**Observation period**:
The STAC `start_datetime`/`end_datetime` of an Item. A label that is exactly
one Item stating it gets it as its cell; a `solar_day` mosaic's cell is the
local solar day; a **composite**'s cell is its resample bin.

**Timespan**:
The inclusive overall period of a region or raster: one start and one end.
_Avoid_: coverage, except ACDD `time_coverage_start`/`time_coverage_end`

**Grid mapping**:
The scalar variable holding CRS parameters (`spatial_ref`), named by each
raster variable's `grid_mapping`. ODC writes it (`gs.write_crs`) and reads it;
`.cf.grid_mappings` also reads it when the CRS has a CF projection name.
_Avoid_: CRS coordinate

**Description attributes**:
`long_name`, `standard_name` and `units`, which say what a variable's values
are (CF chapter 3), read together as the `Description` group.
`.cf.standard_names` and `ds.cf["<standard name>"]` find variables by them.

**Packed data**:
Stored values that decode as `stored * scale_factor + add_offset` (CF §8).
**Stored values** are as written; **decoded values** are physical.
_Avoid_: DN outside Sentinel-2 context, "packing" for the decoded state

**Nodata**:
The stored missing-data value, written as CF `_FillValue` (GDAL and ODC call
it nodata). `missing_value` adds further codes.

**Flags**:
CF `flag_values` or `flag_masks` with `flag_meanings`: what class or bit each
stored value stands for. `scl.cf == "cloud_high"` selects pixels by meaning. A
**legend** is the flags plus a display **palette** (`color_map`), which CF
does not define.

**Declared extent**:
ACDD `geospatial_bounds` with its CRS: the place data describes, independent of
the **grid extent** its pixels cover.

**Collection metadata**:
The STAC Collection and extension URIs a Dataset retains in root attrs.

**Observation properties**:
One STAC property mapping per observation, on the `stac_properties`
coordinate. **Common properties** are their intersection, computed on read.

### Sampling

**Region**:
One place a sample covers: founded by the smallest footprint, joined by every Item covering most of it; a large footprint, such as a radar swath, joins several.
_Avoid_: tile, area, place

**Slot**:
One position on a shared timeline. Each selected collection independently
contributes zero or more observations. With a frequency, slots are calendar
intervals; without one, exact representative timestamps define positions.
See [calendar sampling](docs/concepts/catalog-windows.md).

**Window**:
Consecutive slots of one place taken together as one sample, as pandas `rolling` takes consecutive rows.
_Avoid_: frame, lookback, sequence

**Chip**:
One fixed-size pixel window of a prepared Dataset or DataArray. Chips slices spatial dimensions without choosing a CRS, resolution, or extent; predictions merge back onto the source scene's layout.
_Avoid_: tile, patch, crop

**Composite**:
One Item made from several acquisitions, such as a monthly median. Its coverage
may cross calendar slots; sampling requires an explicit policy for that case.
_Avoid_: mosaic, aggregate
