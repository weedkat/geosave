# GeoSave

GeoSave makes geospatial ML feel like normal Python: native ecosystem objects,
preserved metadata, and lazy execution. The project is Alpha; replace obsolete
designs rather than adding compatibility layers unless the user requires them.

## Architecture and stack

- `src/geosave/geodata/`: raster/vector accessors, metadata, I/O, STAC and
  visualization. Use xarray for rasters, GeoPandas for vectors, PySTAC for STAC,
  and ODC for grids and raster loading.
- `src/geosave/ml/`: model chains, components, processors, native releases
  (Transformers and Hugging Face Hub) and reusable Lightning training support.
- `src/geosave/cli/` and `src/templates/`: workspace generation and its source
  templates. Generated `workspace/` files belong to the user.
- `src/geosave_flows/`: Prefect flows that run a released pipeline through public GeoSave APIs.

Dependencies point one way: `geodata <- ml`, and `geosave <- geosave_flows`.
Geodata imports must not load torch; the inference path of `ml` (chain,
components, processing, release) must not import its training support;
geosave must not import Prefect.

## Working paradigm

- Use native operations first. Before adding a wrapper or workaround, check the
  dependency's current public docs and installed source, try it on the relevant object,
  and reproduce the gap GeoSave must cover.
- Concrete accessors own their variants: DataArray owns one array, Dataset its
  variables, DataTree its nodes. Base classes share identical behavior or declare
  contracts; they do not dispatch by object type. Keep helpers with their owner.
- Preserve CRS, coordinates, transform, nodata, dtype and variable identity.
  Keep supported operations lazy; make pixel reads, writes, reprojection,
  resampling and representation changes explicit.
- Use readable steps, explicit intermediate values and concise names. Prefer
  native objects or typed values; dictionaries suit open-ended metadata.
  Remove duplicate mechanisms; helpers must simplify a named operation rather
  than relocate complexity.
- Preserve unrelated working-tree changes. Edit notebooks only when requested.

## Design and decision questions

For API changes, refactors and dependency or performance work, read
[architecture](docs/contributing/architecture.md) and
[open decisions](docs/contributing/open-decisions.md), then trace the relevant
callers, native objects and consumers. Existing code, tests and agent proposals
are evidence, not automatically agreed requirements.

**Phase 1 — core flow and context:** Show a simplified caller snippet, sample
data shape or diagram covering inputs, outputs, state ownership and connections
to surrounding code. Settle these boundaries before asking about minor edge
cases such as retries, empty-state wording, micro-validations or rare races.
Raise a failure mode here only if it changes the core design.

**Phase 2 — hardening:** Once the core flow and surrounding contracts are
settled, address remaining failure modes, edge cases and polish.

Resolve routine choices from the codebase and existing authorization. When a
user decision is necessary, never ask a bare or open-ended question. Use all
four parts, concisely:

1. **Problem and context:** What existing behavior triggers this decision, and
   why does it matter?
2. **Visual comparison:** Show viable Option A and Option B using simplified
   code, data shapes or a diagram, including their surrounding connections.
3. **Trade-offs:** List concrete pros and cons of each option across complexity,
   coupling, maintainability and UX. Explain consequences in the current workflow,
   not generic labels such as "simpler" or "more flexible". Where a dimension has
   no meaningful difference, say so briefly rather than inventing a trade-off.
4. **Recommendation:** Recommend one option and ask the user to confirm or
   adjust it.

Discussion requests remain discussion. Routine edits without behavior changes
need only relevant context and the bounded edit, not a design review.

## Scoped guidance and verification

Read the relevant [contributor guidance](docs/contributing/agent-reference.md):
[structure](docs/contributing/agent-reference.md#structure) for package/I/O/STAC
work, [ML](docs/contributing/agent-reference.md#machine-learning) for training
and templates, [API docs](docs/contributing/agent-reference.md#public-api-and-docstrings)
for public documentation, and [release](docs/contributing/agent-reference.md#release)
for packaging and publishing. Verify documented maps against source.

`pyproject.toml` owns dependencies and tool configuration; tests mirror source.
Validate changed workflows through observable values, grids, metadata and
execution. Persistence needs round trips; laziness needs execution checks;
performance claims need measurements of the resource at issue. Update obsolete
tests rather than preserving obsolete behavior to satisfy them.

Before finishing, check the requested outcome and update affected docs. Report
changes, checks, breaking changes and remaining risks briefly. Distinguish
failures, expected failures, skipped checks and incomplete runs.
