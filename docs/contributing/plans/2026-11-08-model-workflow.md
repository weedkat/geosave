# Delete old ModelSpec and build version 1

Entry/output selection and construction were subsequently revised in the
[entry/output plan](../plans/2026-11-08-modelspec-entry-output.md). The source/result
examples below record the earlier design.

Status: implementation plan. Production deletion/replacement has not run.
The [revised design](../specs/2026-11-08-model-workflow.md) and
[study](../studies/2026-11-08-modelspec/README.md) describe the candidate.

## Direction

Start from schema version 1. Delete the old implementation and old-schema
integrations. Do not migrate old YAML, retain compatibility methods, or port the
old feature inventory by default. Build only the newly agreed workflow contract.
There is no named-flow registry, generated callable/binding interface, or
bracket/index syntax in `!ref`.

Deleting an implementation also requires deleting its obsolete consumers. This
is dependency cleanup, not a promise to reproduce their behavior in the new
engine. Do not leave broken imports or silently advertise removed features.

## 1. Agree on the minimal new contract

- Invocation inputs, named calls and literal/reference keyword arguments.
- Explicit caller-selected source and result(s).
- Plain references plus existing-object attributes and bound methods.
- Inert load/plan, invocation-local results, and no implicit intermediate override.
- Review input typing/defaults and selected-call binding before expanding them.

## 2. Delete the old implementation

Remove old `model.py`, `call.py`, `stage.py`, `base.py`, `rasters.py`,
`tensors.py`, and `stac.py` implementations under `src/geosave/model/spec/`.
Replace the package exports with the new interface when implementing it.
No compatibility aliases or separate old/new execution modes.

Keep independent useful functions in `model/processing.py` and the native
geodata/STAC/IO implementations. Their usefulness is independent of the obsolete
schema. Do not delete ecosystem operations merely because old recipes call them.

## 3. Delete obsolete integrations

Inventory before deletion, preserving unrelated user edits:

| Location | Cleanup |
| --- | --- |
| `src/geosave/model/release/huggingface.py` | Remove old recipe coupling; add new serialization only against the new contract |
| `src/geosave_flows/ingest.py`, CLI and exports | Remove calls to old ingest/load methods and obsolete commands; keep any new Prefect entrypoint as ordinary Python |
| Six template `model_spec.yaml` files | Delete obsolete recipes; create new examples from scratch where justified |
| Template Datasets/DataModules and local supervised copies | Remove obsolete stage/pixels integrations; do not leave training imports broken |
| Template publication and preparation scripts | Remove stale calls and references to deleted files |
| `tests/conftest.py`, `tests/model/spec`, flow/release/template tests | Remove old fixtures and interface-specific tests; write new behavior tests for the new scope |
| Live guides, concepts, references, README and GLOSSARY | Remove obsolete instructions and document only working new behavior |

If removing an old integration removes a user-facing capability, say so rather
than introducing a migration layer. Rebuild a needed workflow explicitly against
version 1; there is no obligation to preserve old acquisition/configuration
wrappers or the fixed training lifecycle.

Historical documents may retain clearly marked historical names. Notebook and
user-generated workspace editing are excluded unless requested. Report those
exclusions rather than claiming every textual mention has disappeared.

## 4. Implement the replacement cleanly

- New small reference parser; no import of old `Ref`, `CallSpec` or `StageSpec`.
- Strict YAML mapping/tag validation and schema version 1 only.
- Dependency planning from selected source/result, including cycle detection.
- Path-specific input requirements/defaults and preflight before avoidable effects.
- Native results, useful exception context, shared work within one invocation.
- Recipe load/save round-trip using existing `Location`/storage infrastructure.
- Ordinary validation/processing calls; Python owns iteration and resource lifetime.

Do not copy the study wholesale. Its purpose is to test semantics, not establish
the final internal structure or substitute for production persistence checks.

## 5. Build new integrations and examples

- One complete raster-to-result example and an explicit save-result path.
- Acquisition options implemented as ordinary Python, with grid/time policies
  made explicit. Do not reproduce every deleted old recipe automatically.
- An ordinary typed Prefect function calling `spec.run(source=..., result=...)`.
  No callable factory, dynamic signature generation, or spec-owned flow object.
- New release serialization with pinned recipe/code/environment relationships.
- Rebuild selected training examples with project-owned augmentation and shared
  deterministic numerical functions, without the old fixed-stage interface.

## 6. Verify the clean result

- Run new parser/graph tests, native raster laziness/value/grid checks and
  persistence round-trips.
- Check remaining source imports, retained template generation and packaging,
  new release loading, and actual Prefect worker reconstruction.
- Verify removed examples/commands are not still advertised.
- Search active source/templates/tests/docs for obsolete imports, stage method
  assumptions and old schema documents. Assess historical references separately.
- Report exact failures and unverified scientific/GPU/distributed behavior.

No broad compatibility, old-schema conversion, YAML control-flow language,
automatic code installation, sandbox, or persistent raster cache is in scope.

## Implementation evidence — 2026-11-08

Version 1 is implemented in `model/spec/model.py` and `reference.py`. The old
fixed-stage engine and acquisition CLI are deleted. Generated recipes use
project preparation functions and explicit DataModule pixel names. The prototype
engine is retired; its behavioral checks now exercise production code.

Validation on the working tree:

- 140 focused model, release, processing, and template tests passed; three slow
  tests were excluded by the default markers.
- Two distributed training tests passed with local socket access.
- Built-wheel recipe generation passed using the existing offline build cache.
- The preparation walkthrough and five import-layering tests passed.
- A real local Prefect flow completed and preserved saved raster values/grid.
- Scoped Ruff, formatting, type checking, and generated reference freshness passed.

The full default suite was attempted, then interrupted after stalling. It had
reported two failures: the generated preparation script's import path (fixed and
retested) and a notebook test (not investigated; notebooks are outside this task).
No full-suite pass is claimed. Remote worker deployment, custom-code packaging,
and large-scene resource behavior remain unverified. Existing workspaces are
not rewritten or migrated. Execution requires trusted installed Python code.
