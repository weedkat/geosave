# GeoSave

GeoSave connects native geospatial data with model development. Catalogs index data; they are optional for processing a raster.

## Language

### Data and models

**Catalog**: A table of geospatial Items and their assets, used to discover and
select data. It is not a model input tensor.

**Item**: One STAC observation. `RasterStac.from_dataset` groups an existing
Item dimension (`time` by default); each group becomes one Item.
_Avoid_: observation row, inventory row

**Asset**: One file within an Item. Raster publication writes one COG per
variable and Item, named `<root>/<item id>/<variable>.tif`.
_Avoid_: inventory, layer

**Sample**: The inputs belonging to one model example, with targets when labelled.
A sample may contain one observation or a temporal window; it need not be a catalog row.

**Scene**: Imagery associated with one observation. This does not imply one file,
one grid, or one model-sized chip. Use explicit dimensions and grids to describe
what a model accepts.

**Grid**: For an affine raster, the CRS, affine transform and array shape that
place its pixels. Resolution alone does not identify a grid. General DataTrees
have caller-defined nodes; `RasterStac` separately owns the publication layout.

**Recipe**: The former ModelSpec YAML declaration of processing calls.
ModelSpec is disabled. Registered processing roles now describe ingestion,
preprocessing and postprocessing; release integration is still open. See
[release and inference](docs/contributing/design/release-inference.md).

### Raster structure

Terms follow CF, xarray and cf-xarray. See
[raster structure](docs/concepts/xarray-structure.md).

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
STAC values stored as coordinates marked `stac_property`. The accessor's
`stac_properties` mapping identifies each STAC key's coordinate. Loading retains
properties requested with `with_properties`.

### Sampling

**Region**:
A geometry and optional time interval, independent of pixel placement.

**Slot**:
One calendar interval in raster temporal sampling. First/last selection keeps
one observation per slot; an empty slot excludes a window.

**Window**:
Consecutive calendar slots of a prepared raster, returned by `.gs.sample`.

**Chip**:
One fixed-size pixel window of a prepared raster. `RasterChipper` cuts chips along the spatial dimensions without choosing a CRS, resolution, or extent; `RasterMerger` places predictions back by chip index.
_Avoid_: tile, patch, crop

**Composite**:
One Item made from several acquisitions, such as a monthly median. Its coverage
may cross calendar slots; sampling requires an explicit policy for that case.
_Avoid_: mosaic, aggregate
