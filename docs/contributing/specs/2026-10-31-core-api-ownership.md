# Core API and vector accessor ownership

Status: implemented on 2026-10-31. Verification is recorded in the implementation plan.

## Caller experience

```python
import geosave as gs

band = gs.create_dataarray(pixels, grid, dims=("y", "x"))
scene = gs.create_dataset({"red": band})
inputs = gs.create_datatree({"image": scene})
catalog = gs.create_catalog(items, collections=[collection])
samples = catalog.gs.sample()
chips = samples.gs.chip(224, target=grid)
data = chips.iloc[0].gs.to_xarray()

plots = gs.GeoDataFrame({"class": [1]}, geometry=[polygon], crs="EPSG:4326")
mask = plots.gs.rasterize(scene, column="class")
regions = mask.gs.vectorize()
```

Values remain native xarray and GeoPandas objects. GeoSave classes are
accessors except for existing value objects such as GeoAnchor. There is no
create_vector factory: native GeoDataFrame construction suffices.

## Intent and constraints

Make construction, accessor registration, domain behavior, STAC translation,
and file serialization have explicit owners. Preserve lazy raster behavior,
metadata, grid semantics, sample recipes, storage options and current formats.
Keep the raster and vector accessor families. Do not flatten core or create
new wrapper containers. Use existing dependencies; do not change version floors.
Do not change ModelSpec or sampling algorithms. Ignore notebooks in this work.
Preserve unrelated working-tree changes. No compatibility aliases are required.

The user accepted create_* factories and a common parent for catalog and sample
accessors. The user authorized the proposed internal parent and layout by requesting implementation.

## Upstream evidence

The installed dependency and lockfile use stac-geoparquet 0.8.2. Its public
Arrow API converts Items to RecordBatchReader/Table and back, and writes
Parquet. It does not supply this project's accessor hierarchy. Its useful
architectural precedent is separating row records, table metadata and conversion.

- [Usage](https://stac-utils.github.io/stac-geoparquet/latest/usage/): Arrow is
  the interchange representation between formats.
- [Arrow API](https://stac-utils.github.io/stac-geoparquet/latest/api/arrow/):
  parse_stac_items_to_arrow, stac_table_to_items and to_parquet are public.
- [Specification](https://github.com/radiantearth/stac-geoparquet-spec/blob/main/stac-geoparquet-spec.md):
  each row represents one Item; properties become columns; Collection definitions
  are a separate mapping in file metadata. The JSON field type is optional.

Geometry is one feature attribute, not necessarily a boundary: points, lines,
polygons and multipart geometries are vector data. GeoCatalog and GeoSamples
both describe vector tables even when their referenced assets contain pixels.
An ordinary GeoDataFrame does not require STAC fields or Collection metadata.

## Approaches

1. **One schema-switching accessor.** Simplifies class registration but exposes
   Item and sample operations on tables that cannot satisfy them. Not recommended.
2. **GeoSamples inherits GeoCatalog.** Shares implementation but falsely implies
   each sample row satisfies Item-table operations and persistence. Not recommended.
3. **Sibling schemas under a STAC-metadata parent.** Approved approach. Share only
   the attached Collection metadata contract and general vector operations.
   Preserve distinct Item and sample schemas and writers.

## Accessor hierarchy

```text
GeoVector                         ordinary GeoDataFrame integration
└── StacMetadata                  internal, attached STAC Collection definitions
    ├── GeoCatalog                one Item per row
    └── GeoSamples                one input example per row

GeoCatalogRow                           one catalog row -> Dataset
GeoSample                         one sample row -> DataTree
```

StacMetadata is the internal replacement for CollectionTable. It is
neither a table of Collections nor a writer. Do not name it StacTable: that name
already belongs to the STAC GeoParquet search client in stac/search/table.py.
Do not export the internal parent from core or the package root.

GeoVector owns GeoSave integration with a GeoDataFrame: spatial/temporal query,
exact-grid rasterization, GeoSave visualization and path-based I/O. Retain the
current public methods during this migration except from_geometry. Do not remove
methods simply because GeoPandas has a similarly named operation; first compare
CRS behavior, storage options, return values and metadata guarantees.

StacMetadata owns collections, Collection-preserving concat, and the existing
refusal of GeoJSON/GeoPackage exports that lose the required definitions. Keep
the existing first-definition-wins concat policy and independent copies. It
does not own Item conversion, sample planning, or file metadata encoding.
GeoVector.concat continues to own geometry normalization, kind/CRS checks and
native row concatenation. Do not introduce a configurable general table engine.

GeoCatalog owns Item-table operations and delegates STAC conversion and writes.
GeoSamples owns sample-table operations and delegates frame/chip/merge planning.
Row accessors interpret their respective row schemas and open assets lazily.
Shared to_xarray spelling is appropriate despite different return structures.
Do not introduce table-level to_xarray or arbitrary row-merging semantics.

## Ownership and files

```text
geodata/core/
  __init__.py
  anchor.py
  register.py                     all xarray and pandas registration
  constructors/
    __init__.py
    dataarray.py                  create_dataarray, delegates to create_dataset
    dataset.py                    create_dataset, variable metadata inheritance
    datatree.py                   create_datatree, groups and shared root grid
    catalog.py                    create_catalog, delegates to STAC translator
  raster/
    __init__.py
    base.py                       existing shared GeoRaster behavior
    grid.py                       reference coordinates and CF axis metadata
    dataarray.py                  GeoDataArray
    dataset.py                    GeoDataset
    datatree.py                   GeoDataTree and existing group operations
  vector/
    __init__.py
    vector.py                     GeoVector
    stac.py                       internal StacMetadata
    catalog.py                    GeoCatalog and GeoCatalogRow
    samples.py                    GeoSamples and GeoSample
    schema.py                     sample fields and persisted frame-recipe types
    query.py                      shared spatial/temporal selection
```

All .gs registrations move to core/register.py. Xarray registrations bind their
classes directly; pandas keeps table/row schema dispatch. Import registration
once after class definitions through core/__init__.py. Importing geosave must
still register accessors without loading torch or the visualization stack.

Merge item.py into catalog.py and sample.py into samples.py. Move the sample
schema field tuples and _TimeBin/_FrameRecipe TypedDicts into vector/schema.py;
both planners and readers import the same definitions. Keep Window and
WindowFunction where their current public types are owned by GeoSamples.
Do not add a row base until shared behavior actually requires one.

The constructor signatures preserve current arguments and return types, with
only these function renames: array -> create_dataarray, raster -> create_dataset,
tree -> create_datatree. Move create_catalog out of stac/create.py. Native STAC
create_collection and create_items remain there. Move RasterVariable and the
reference type beside create_dataset. The dataarray constructor imports the
reference type from that module; do not duplicate constructor implementations.

Grid reference extraction, coordinate assignment and write_cf_axes live in
raster/grid.py. Dataset construction and GeoDataset.write_crs use those helpers.
Band metadata inheritance stays in constructors/dataset.py. Preserve GCP grids,
exact coordinate joins, existing inheritance and explicitly supplied overrides.

Root exports contain the four create_* factories and all public accessor classes,
including GeoCatalogRow and GeoSample. Remove the old factories and
gs.stac.create_catalog; do not retain forwarding aliases. GeoVector.from_geometry
is removed; callers use native GeoDataFrame and existing geometry parsing helpers
where non-Shapely input requires conversion. Retain GeoVector's other methods.
Update code, tests, generated templates and text documentation. Do not edit ipynb.

## Conversion and persistence boundaries

stac/translate/table.py owns Item <-> native table conversion, using upstream
public conversion APIs wherever they preserve our behavior. io/vector owns
storage options, relative href handling and file metadata encoding/restoration.
Keep native Collection copies in accessor results, with dictionaries stored in
GeoDataFrame.attrs["stac:collections"].

Catalogs remain STAC GeoParquet, with Collection definitions under
stac-geoparquet/collections. Samples remain ordinary GeoParquet using the
geosave:collections metadata key. Samples are not encoded as fake STAC Items.
Do not change JSON sample payloads to Arrow structs during this API migration.

Evaluate stac_table_to_items against the existing rustac conversion with a
focused smoke. Replace rustac conversion and its bbox/chunk workarounds only if
filtered/concatenated/empty and heterogeneous-asset cases preserve semantics.
Otherwise retain the current converter and record the specific mismatch.

Schema recognition is distinct from validation. Pin compatibility with native
upstream files and test the optional JSON-field case before changing detection.
Use format metadata plus canonical Item fields at the file reader boundary to
identify STAC tables; normalize omitted JSON-only fields there if conversion
requires them. Do not classify arbitrary business tables from id/assets alone.
Preserve recognition of in-memory tables produced by create_catalog, empty
results, and persisted samples. Record the exact normalized fields in tests.

## Validation and risks

Baseline on 2026-10-31: 251 passed, one failed across vector accessors, Item/table
translation, STAC GeoParquet and raster constructors. The failing validation
test could not resolve stac-extensions.github.io for the projection schema;
it is an external-schema fetch failure, not evidence of a green full suite.

An upstream smoke initially used an assetless Item; PyArrow rejected an empty
assets struct. Repeat with a real asset reference; do not add a dummy production
asset or change the API to accommodate the probe fixture.

The corrected smoke passed: a file written directly by upstream to_parquet
was read as GeoCatalog, with its Collection restored. Omitting type and
stac_version while keeping STAC file metadata reproduced a compatibility gap:
the current reader selected GeoVector and restored no Collection definitions.
This finding motivates the reader normalization task; it is not a reason to
make every id/assets business table a catalog.

A second smoke tried upstream stac_table_to_items on Arrow exported directly
from GeoSave's GeoDataFrame. Full and filtered tables raised "Expected 4 or 6
fields in bbox struct"; the empty table returned no Items. Our translator drops
the covering bbox column on import, so swapping converters alone is insufficient.
Retain the current working converter until the prescribed complete comparison
demonstrates an actual simplification with identical behavior.

Meaningful checks: upstream file interoperability; all registration paths;
native return types and lazy data identity; Item and sample persistence;
relative paths; mixed grids; frame/chip order; metadata concat and copying;
first-definition precedence; source-mirrored tests; template generation;
root import without torch/viz; lint/type checks and generated API freshness.

Public breaks are factory/import renames and removal of from_geometry. Notebook
examples may retain obsolete calls because notebooks are outside this task.
No serialization migration is planned. Import cycles, schema ambiguity and
metadata loss are the main implementation risks.
