# GeoSave

GeoSave makes geospatial ML feel like normal Python through native ecosystem
objects, preserved metadata and lazy execution. The project is in Alpha:
replace obsolete designs rather than adding compatibility layers unless the
user requires compatibility. Prioritize a sound design over a fast patch.

## Before changing behavior

For API changes, refactors, dependency integration and performance work:

1. Establish the user's workflow: inputs, desired output and why the operation
   exists. Read the relevant conversation, `docs/contributing/architecture.md`
   and `docs/contributing/open-decisions.md`. Separate agreed requirements from
   agent proposals.
2. Trace the operation through callers, readers, native objects and consumers.
   Inspect adjacent variants and identify who owns data, metadata and execution.
   A referenced method is an entry point, not the boundary of the investigation.
3. Check the dependency's public API, supported object types, parameters and
   defaults in current documentation and installed source. Try the direct
   native operation on the user's object in a throwaway spike before designing
   a wrapper. Reproduce any limitation that would require a workaround.
4. Review parameters, defaults, return values and caller needs together.
   Legacy code and tests are evidence, not automatic requirements. Retain custom
   behavior for an agreed use case or demonstrated correctness requirement.
5. Before implementation, briefly show the proposed caller code, the verified
   gap GeoSave must cover, and what existing machinery can be removed. Resolve
   uncertainties that affect the design; use existing authorization rather
   than repeatedly asking permission. Discussion requests remain discussion.

For a routine edit with no behavior change, inspect the relevant context and
make the bounded change; the full investigation above is unnecessary.

If implementation grows preparation/restoration pipelines, variant branches
or helpers that merely relocate complexity, revisit the design. Test outcomes
independently of the code shape. Passing tests alone do not validate an API.
For performance claims, measure the resource at issue: small output and short
runtime do not prove fewer source reads. Report evidence and conclusions
briefly, without an internal reasoning transcript.

## Implementation principles

- Prefer xarray, GeoPandas, PySTAC, ODC and Lightning capabilities over parallel
  GeoSave abstractions. A wrapper must own a necessary invariant the native
  object cannot express. A dependency's internal loop does not justify our own.
- Keep variant-specific behavior in its concrete accessor. DataArray owns one
  array, Dataset its variables, and DataTree its nodes; use native Dataset/tree
  operations where they already provide the required behavior. Base classes
  may declare contracts or share identical behavior, not dispatch by object type.
- Simplify by removing unnecessary cases and duplicate mechanisms. Keep required
  correctness. Helpers should own a named operation with real reuse or clarity.
  Some repetition is preferable to a harder-to-follow abstraction.
- Use explicit loops and intermediate values; no nested comprehensions or clever
  one-liners. Group each meaningful step with blank lines. Use concise names:
  verbs for functions, values for variables. Tutorials follow the same standard.
- Prefer native objects or typed values over dictionaries whose keys encode
  structure. Dictionaries suit open-ended attrs/properties and dependency APIs.
  Union types beyond optional values deserve scrutiny: retain them when the
  behavior is genuinely the same, including native API contracts.
- Preserve applicable CRS, coordinates, spatial dimensions, transform, nodata,
  dtype and variable identity. Keep operations lazy where possible. Reprojection,
  resampling, dtype/coordinate changes and eager computation are explicit.
- Keep unrelated code and working-tree changes intact. Add no compatibility
  aliases, duplicate execution paths or dependencies without a concrete need.
  Ignore notebooks unless the user asks to work on them.

## Navigation and scoped guidance

Read the linked sections before working in their scope:

- Package boundaries, I/O, STAC or file placement:
  [Structure](docs/contributing/agent-reference.md#structure) and
  [architecture](docs/contributing/architecture.md). Verify maps against source.
- Model, training, CLI-generated workspaces or templates:
  [Machine Learning](docs/contributing/agent-reference.md#machine-learning).
  `src/templates/` is the source; `workspace/` is consumer output.
- Public API documentation:
  [Public API and Docstrings](docs/contributing/agent-reference.md#public-api-and-docstrings).
  Use concise Google-style docstrings and native-object examples.
- Packaging, publishing or documentation builds:
  [Release](docs/contributing/agent-reference.md#release).

Dependencies point one way: `geodata <- model <- ml`, and
`geosave <- geosave_flows`. Keep torch out of geodata imports, ML out of model,
and Prefect out of geosave. Internal helpers stay with their owning package;
supported public API is declared explicitly.

## Validation

`pyproject.toml` owns dependency and tool configuration. Tests mirror source.

```bash
uv sync                         # dev group includes geosave[all]
uv run pytest tests/path/to/test_file.py
uv run pytest                   # full suite
uv run pytest -m integration
uv run pytest -m slow
uv run ruff check .
uv run pre-commit run --all-files
```

Choose checks from changed user workflows and meaningful failure modes: wrong
values/grids, lost metadata, eager reads, broken training or unusable artifacts.
Persistence changes need round trips; laziness promises need execution checks.
Each test must add distinct protection. Avoid tests for private structure,
removed names, incidental defaults, exact export lists or dependency validation.
Update tests for intentional contract changes rather than preserving obsolete
behavior to satisfy them. Distinguish new failures, known failures, expected
failures, skipped checks and incomplete runs when reporting results.

## Communication and completion

Show caller code or a small printed object before explaining a design. Scale
explanation to the user's question; lengthy technical detail is not evidence
of understanding. Keep obvious accessors short and comments about domain reasons.

Before finishing, review the change against the requested outcome, including
whether responsibilities and complexity improved. Update affected plans/docs.
Report what changed, checks run, and breaking changes or remaining risks.
