# Core vector cleanup

Implement the approved vector review in the existing checkout, preserving the
uncommitted refactor it builds on. No compatibility aliases or notebook edits.

1. Add regressions for sample concat metadata, unsupported exports, renamed
   geometry, and native-grid padding across CRSs. Observe failures first.
2. Use the existing raster anchor for affine grids. Let STAC loading apply
   Collection ACDD, so item and sample row readers share its implementation.
3. Compute padding on the native pixel lattice from the sample ground extent.
   Keep pixels lazy and preserve exact same-grid reads.
4. Preserve referenced Collections on sample concat, reject conflicting
   definitions, and reject exports without the sample metadata contract.
5. Recognize samples by their data columns independently of geometry spelling.
   Keep the storage column order and caller-added columns.
6. Rename dataframe.py to accessors.py and the internal catalog package/tests
   to sampling. Update live imports, documentation, and stale stack wording.
7. Run affected tests and lint; review changes and record remaining limitations.

Validation: vector, sampling, STAC loading, vector I/O, model/spec and layering
checks. Include Collection round trips and lazy native/aligned chip reads.

Progress: reviewed existing implementation and initial smokes. Baseline vector
suite had 77 passes and one external schema DNS failure. Work proceeds in place
because the approved refactor is uncommitted in this checkout.

Completed:
- Added failing regressions for metadata loss, export rejection, and geometry
  recognition. Replayed the old padding method against the mixed-CRS regression;
  it requested 1,200,001 pixels of padding and failed the allocation guard.
- Applied Collection metadata in STAC loading; removed both vector helpers.
- Preserved native grids and lazy data while calculating padding from ground.
- Preserved nested Collections on concat and GeoParquet round trips; rejected
  conflicting definitions and sample GeoJSON/GeoPackage exports.
- Separated sample identifying fields from storage columns. Review found that
  frame/chip construction also needed to carry the active geometry name; added
  failing regressions and fixed that propagation, including frame CRS.
- Renamed internal catalog/tests to sampling and dataframe.py to accessors.py;
  refreshed generated references and relevant documentation.
- A real timeless fixture exposed an existing sentinel timestamp overflow in
  STAC loading. Reused Item instant interpretation and added a lazy Item/sample
  reading regression; it now passes without artificial time coordinates.

Final validation:
- 503 passed, 7 deselected across core/vector, sampling, STAC, vector I/O,
  model/spec, and dependency layering. External schema validation passed with
  network access; the original restricted runs could not resolve schema hosts.
- Focused BasedPyright: zero errors/warnings across core/vector, sampling,
  stac/load.py, and stac/translate/collection.py.
- Focused Ruff lint/format, generated API freshness, and git diff whitespace
  checks passed. Independent review found no further issues after geometry fixes.
- Full repository tests and notebook execution were outside this validation.

Further vector refinement:
- Removed `stac.translate.collection.merge_definitions`: it implemented table
  metadata bookkeeping, not STAC translation. `core/vector/table.py` now owns
  the shared Collection decoding, concat conflict checks/deep copies, and lossy
  export restrictions. Catalog/sample accessors only define referenced ids.
- Exposed stored sample grid/span/padding as `geobox`, `timespan`, and
  `pad_widths`. Sampling reconstruction reads these directly. Query argument
  resolution remains separate from stored row dates; their contracts differ.
- Shared table contract smokes passed before restructuring. New sample property
  tests failed before implementation and passed after it. Independent review
  checked mixed-kind concatenation and inherited documentation without findings.
- Refinement validation: 515 passed, 7 deselected in the affected suite;
  focused BasedPyright reported zero errors/warnings; Ruff lint/format,
  generated-reference freshness, and whitespace checks passed.

Query and sample reading ownership refinement:
- Moved complete vector query interpretation to `vector/query.py` and removed
  the parent accessor's private argument readers.
- Sample reading now uses `transform.temporal.select` and `extent.crop`;
  removed bespoke temporal selection and cropping. The reader is named
  `_read_raster` for its return value. Added spacing between logical steps.
- Temporal selection preserves lazy pixels, composite bounds (including scalar
  bounds), source order, and static tree groups. Collection selection validation
  stays in the sample's public reading method.
- Kept `GeoSamples`: proposed replacement names were rejected and `GeoDataset`
  already names the xarray accessor. No raster naming change was authorized.
- Validation: 795 passed, 7 deselected, 3 expected failures across vector,
  sampling, transforms, STAC, vector I/O, model/spec, and layering. After the
  private reader rename, all 12 sample tests passed again. Focused BasedPyright,
  Ruff lint/format, API freshness, and whitespace checks passed. Independent
  review found no blocking issues.

STAC reading ownership refinement:
- Removed `stac/load.py`. ODC loading, interval fusion, configured band mapping,
  and source metadata agreement now belong to `StacSource`.
- Moved native Item asset opening to `io/raster/stac.open_raster`. Vector row
  readers retain their public interface and call this I/O implementation.
- Shared positional band renaming with saved-file assembly, preserving each
  caller's fallback names. Renamed assembly's `_stated` to `_apply_header`.
- Shared published band metadata interpretation in `translate/band.read_attrs`
  and moved observation dates into `translate/item.assign_time`. File/loader
  metadata precedence and the two loading paths' conflict policies stay intact.
- Split loading into meaningful steps with paragraph spacing. Moved metadata
  tests beside the source owner and added native asset-reading regressions.
- Kept native variable-name resolution: ODC's parsing/filtering contract is not
  interchangeable with native Item metadata interpretation.
- Validation: 674 passed, 7 deselected across vector, sampling, STAC, raster and
  vector I/O, model/spec, and layering with public schema network access.
  Focused BasedPyright reported zero errors/warnings; Ruff lint/format, generated
  API freshness, and whitespace checks passed. The restricted-network suite was
  stopped after failures and replaced by the successful network-enabled run.
- Breaking internal imports: `stac.load` was removed without compatibility
  aliases. Historical design documents retain their original module names.

Sampling ownership and reconstruction refinement:
- Moved table orchestration into `GeoCatalog.join` and `GeoSamples.frame`,
  `chip`, and `merge`. `GeoSample` owns native row construction and updates;
  sampling retains footprint, calendar, tiling, and numerical algorithms.
- Removed `sampling/items.py`, flatten/regroup work in framing, and redundant
  Item dict round trips in joining. Frame selection parses Item spans once per
  sample. Pandas now places the period boundaries, including hourly frames.
- Chips persist a parent spatial key; merge also matches the current frame span.
  Nested reconstruction works after persistence, concatenation, and reordering.
  Merge rejects ambiguous labels/parents, incompatible pixel lattices, and
  missing validity masks. Join uses the active geometry column.
- Preserved the user's simplified Collection concat policy: first definition
  per id, including definitions retained after filtering. Updated two stale
  publication tests and guides that still expected conflict rejection.
- Verified and removed stale root `core/dataarray.py` and `core/dataset.py`
  copies, including their staged entries. They differed from `core/raster/`
  implementations only in obsolete imports; live exports use `core/raster/`.
- Nine new regressions failed before implementation and passed afterward.
  Additional checks cover frame/chip ordering and ambiguous parent identities.
  Final affected suite: 527 passed, 7 deselected. Focused BasedPyright, Ruff
  lint/format, API-reference freshness, and whitespace checks passed.
- Older chip tables without parent references remain readable but must be
  regenerated before merging. No compatibility path guesses their parents.

Reference sampling and ownership correction:
- Replaced `GeoCatalog.join(on=...)` with `sample(reference=...)`, without an
  alias. A single collection infers its reference; multiple collections require
  an explicit choice. Each reference Item creates one sample with its own grid,
  footprint, and time span, even when footprints and dates repeat.
- Removed implicit footprint grouping and span aggregation. Other collections
  retain spatially covering candidates for calendar pairing; sampling omits
  missing collections and warns on partial coverage. Union/intersection and
  required-source policies remain outside this change.
- Corrected the earlier orchestration move: core accessors now delegate table
  operations to `sampling`. `GeoSample` still owns row construction and updates.
  Raster reading and grid alignment remain explicit operations.
- Migrated source callers, tests, templates, and active documentation. Historical
  design documents and notebooks were left unchanged. ModelSpec.prepare keeps
  its existing `on` argument and passes it as the sampling reference.
- Added regressions for inferred references, ambiguous multi-collection input,
  reference-owned dates, missing spatial coverage, and lazy single-source chip
  reading. Updated sequence tests to use an actual spanning reference and made
  frame/chip reconstruction assert nonempty parents. Relocation reads every
  optical observation independently instead of expecting implicit grouping.
- Validation: 531 passed, 7 deselected across core/vector, sampling, STAC,
  vector I/O, model/spec, and layering with public schema access. Focused
  BasedPyright reported zero errors/warnings. Ruff lint/format, generated API
  freshness, and whitespace checks passed. The single-source chip smoke also
  passed independently. This is affected-suite coverage, not a full-repo run.

STAC asset and search ownership follow-up:
- Item translation owns eligible data-asset selection, shared by native reading,
  sample-grid selection, and band naming. Explicit non-data assets no longer
  influence grids or reserve variable names; unstated roles remain eligible.
- Kept native naming as an explicit contract independent of ODC configuration.
  ODC resolution and execution remain in StacSource. Multiple eligible grid
  resolutions still select the first gridded asset; explicit choice is in TODO.
- StacClient.search attaches copied definitions for returned Collections.
  StacTable inherits that implementation and StacSource.search delegates to it.
  Search results can be published without manually retrieving definitions;
  empty searches fetch none. search_items remains Item-only.
- Table translation owns optional Collection attachment and Item identity
  validation. Removed its GeoCatalog import and the writers' dependency on the
  public builder module. No compatibility alias retains create.validate_items.
- Corrected source.load's documented conflict behavior: warn and omit disputed
  source attrs, preserving authoritative loader metadata and digital numbers.
- Regression checks failed before implementation for thumbnail-selected grids
  and missing search Collection definitions. Tests cover native lazy reading,
  pixel preservation, search/save/reopen, metadata isolation, and empty searches.
- Validation: affected suite 537 passed, 7 deselected. After consolidating the
  source search path, 28 source/native-reader tests passed again. Focused
  BasedPyright reported zero errors/warnings; Ruff, generated API freshness,
  and whitespace checks passed. Notebooks and unrelated edits were preserved.

StacSource responsibility reduction (supersedes automatic temporal restoration):
- Source loading now searches, calls odc.stac.load with configured options, and
  applies agreed raster-band/Collection metadata plus spatial conventions.
- Removed interval injection, native Item copying, custom date fusers, interval
  conversion, automatic property-coordinate promotion, and max_items squeezing.
  Removed the unused EO coordinate-description helper with its sole caller.
- ODC owns observation grouping, dates, and requested property variables. Metadata
  application skips nonspatial property variables and preserves their native attrs.
- Native Item reading retains its own interval contract. Source-load publication
  no longer promises composite interval reconstruction. Explicitly set properties
  as coordinates when publishing them, and retain original Items for provenance.
- Six initial contract checks failed under the old behavior. Direct comparisons
  now cover singleton output, mixed instant/composite Items, id/time grouping,
  ordinary properties, and explicitly requested interval strings. Metadata tests
  use spatial raster bands instead of one-dimensional stand-ins.
- Validation: 899 passed, 25 deselected across STAC, I/O, core/vector, sampling,
  features, model/spec, and layering. Native ODC parity cases passed for all six
  grouping/property combinations. Focused BasedPyright, Ruff, API freshness, and
  whitespace checks passed. The broad run exited zero but printed an uninformative
  sys.excepthook diagnostic after pytest's summary; no test failure was reported.

Translation family ownership and naming refinement:
- Reviewed all six translate modules. Kept STAC-level composition, with public
  operations before private implementation details and spacing between stages.
- Asset.from_file returns a native Asset; Item assembly calls PySTAC.add_asset.
  The owner argument lets extension translators declare schemas through PySTAC.
  Removed the hidden asset-attachment operation and single-use media-type helper.
- Band translation has one attrs reader and one metadata writer. Removed stamp,
  made schema-field merging private, and kept published aliases for native band
  naming. No fabricated raster or new wrapper represents translated metadata.
- Item readers now use read_* names. Publication separates identity, scalar
  properties, nominal observation time, and coverage. Reads coordinates directly
  instead of constructing an xarray Dataset solely to access dates. TimeSpec owns
  calendar bounds. The footprint-centroid identity rule stays unchanged; it differs
  from GeoAnchor's transformed grid centroid and cannot safely reuse that formatter.
- Coverage now includes every file's explicit composite bounds and normalizes
  calendar/instant spans before combining them. Regression tests cover both gaps.
- Collection.copy_for_items names the actual copying/publication contract and groups
  Items once. PySTAC still derives extents and resolves hrefs. Table conversion uses
  Arrow/rustac directly, without the single-use Arrow helper. validate_ids explicitly
  names its identity check; it is not STAC schema validation.
- All affected source/test callers and generated references updated without aliases.
- Verification: 757 passed, 20 deselected across STAC, I/O, core/vector, sampling,
  and model/spec; 75 passed, 1 deselected in the focused final publication/translation
  run after coverage consolidation. Focused Ruff and BasedPyright passed. The broad
  run again exited zero with uninformative sys.excepthook messages after its summary.

Follow-up Collection and missing-grid ownership corrections:
- Collection.read_attrs returns ordinary root attrs, matching the band reader.
  Removed the special to_acdd return contract. Metadata application uses xarray
  assign_attrs, preserving existing root fields absent from the Collection.
- Moved Collection preparation out of translation into io/vector/_stac.py for the
  two publication writers. Tried native Collection.clone; an existing API-client
  publication smoke failed because CollectionClient requires its root. Retained
  direct to_dict/from_dict conversion to native values; no fallback or adapter.
- read_grid now warns and returns None when no eligible asset states a grid/CRS.
  GeoItem.anchor is optional; sampling warns and skips missing-grid references.
  Native image reading is unchanged and tested lazy with unchanged pixels/no CRS.
- Local ODC probe: missing Item/file georeferencing failed automatic grid inference;
  supplying an output grid also failed. Native reading succeeded, so no ODC image
  fallback or invented geographic bounds were added.
- Final validation: 760 passed, 20 deselected across STAC, I/O, vector, sampling
  and model/spec; two focused missing-grid cases also passed, including a mixed
  catalog retaining valid references. Scoped BasedPyright reported zero errors
  and warnings; Ruff, generated API freshness, and whitespace checks passed.
  The broad suite exited zero but still printed sys.excepthook teardown diagnostics.
