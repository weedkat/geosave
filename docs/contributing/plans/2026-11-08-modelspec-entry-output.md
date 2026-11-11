# ModelSpec entry/output and implementation cleanup

The workflow remains a configurable graph of installed operations. Alternative
entry paths converge through a selected reference; output branches share step
results. No additional raster contract checker or YAML type restriction is added.

## Implementation

1. Use `entry` and `output` as optional saved defaults and run/plan overrides.
   Every run resolves both selections. An entry may have prerequisites; it must
   participate in the selected output's dependencies. Use `!ref entry` for the
   selected connection. Reserve selector names and use `destination` for files.
2. Construct ModelSpec from input/step declarations. `from_dict` translates the
   saved format, `from_yaml` parses text, and `load` reads storage. Save from the
   authoritative declarations rather than retaining a second source document.
3. Move YAML constructors and serialization to `spec/yaml.py`. Keep native
   reference handling separate. Keep planning/execution owned by ModelSpec.
4. Preserve original errors and add outcome, step, operation, argument, and
   reference context where applicable. Missing inputs explain where they are
   needed and how to supply them. Never include full input values in context.
5. Migrate templates, tests, executable studies, and current guides. Do not edit
   notebooks or generated consumer workspaces. No compatibility aliases.

## Verification

Smoke default selection and explicit overrides before settling the interface.
Exercise lazy raster preparation, shared output branches, storage round trips,
invalid selections before side effects, and actionable failures. Run ModelSpec,
release, processing, and template checks plus focused lint/type checks.

## Breaking changes

`ModelSpec(text)` becomes `ModelSpec.from_yaml(text)`. `source`/`result` selectors
become `entry`/`output`; `!ref source` becomes `!ref entry`. Output-file inputs use
`destination`. Existing schema version 1 is updated during this unreleased Alpha
redesign; old declarations are not adapted automatically.

## Completion

Implemented the declaration constructor, YAML persistence separation, saved
entry/output defaults, connected-entry check, contextual errors, glossary, and
caller migration. Runtime literal handling and YAML type support were not changed.

Validation: 163 tests passed across ModelSpec, processing, releases, templates,
and CLI; 6 slow/integration tests were deselected by repository defaults. Smoke
execution verified default raster prediction, overridden paths, missing-destination
messages, and all six geospatial template defaults. Focused Ruff and basedpyright
checks passed. The full repository suite and Prefect server smoke were not run.

## Declaration and naming follow-up

Input and Step use Pydantic for declaration structure. Input owns primitive value
checking; Step owns operation-path validation and dependency inspection. Following
design review, ModelSpec remains a plain class owning connections, planning,
execution, and persistence. Its constructor accepts validated declarations;
from_dict/from_yaml handle raw saved declarations. YAML parsing remains separate,
with references preserved during saving. ModelSpec does not expose Pydantic's
model_dump/model_copy/model_construct interface.

Reference handling now names its operations explicitly: `read_value`,
`find_references`, and `bind_arguments`. Runtime values use `object`; bound
arguments have a concrete dictionary shape. No `Any` annotations remain in spec.
The guide explains that operation implementations determine concrete output types.

Follow-up validation: 164 scoped tests passed, 6 slow/integration tests deselected;
focused lint and type checks passed. Existing YAML workflow syntax is unchanged.
Python Input/Step declarations are models rather than dictionaries, and
`Reference.resolve` was renamed to `Reference.read_value` without an alias.

Revalidation after restoring plain ModelSpec: 48 focused tests passed; the wider
run excluding release tests passed 151 tests with 5 deselected. Release checks
could not collect because tests/model/stages/encoder/test_prithvi.py imports
TimeSpec, which is absent from geosave.geodata.attrs in the current working tree.
That unrelated failure was left unchanged. Focused Ruff, formatting, basedpyright,
and diff checks passed.

## Redundant-mechanism cleanup

Implemented the agreed review findings:

- Workflow documents contain an explicit `steps` mapping. Loading no longer
  infers steps by excluding metadata; saving writes the same structure. Only
  `entry` is reserved as a step name. Root fields remain checked for typos.
- Execution uses one pass. Imported operations and referenced methods are
  loaded/read, bound, and called when their step executes. Removed signature
  preflight, its function table, and Python-version inspection branches.
- Removed Input.type, the primitive-type vocabulary, and runtime type adapters
  for supplied values. Consuming operations own value validation. Declaration
  structure is still validated by Input/Step, and missing/unused external inputs
  still fail before operations are imported or called.
- Step identifiers and operation paths are checked directly rather than by
  constructing temporary Reference objects. Output selections copy only lists;
  mutable input defaults retain deep copying between runs.

Templates, the current guides, executable study YAML, and relevant test fixtures
were migrated. The unreleased schema version remains 1; old flat step documents
and input type declarations are rejected rather than adapted. Existing notebooks
and generated workspaces were not edited.

Verification: 151 tests passed with 5 deselected in the workflow/processing/
template/CLI run. After adding the nested metadata-name round-trip regression,
all 49 focused ModelSpec tests passed. All six template defaults plan correctly.
Focused lint, formatting, type checks, and diff checks passed. Release test
collection remains blocked by its unrelated TimeSpec import; the full repository
suite and Prefect server smoke were not run.

Changed failure timing: an operation import or invalid call can now fail after
its dependencies have executed. Python supplies the original error and ModelSpec
adds step/output context. Prior side effects are not rolled back.
