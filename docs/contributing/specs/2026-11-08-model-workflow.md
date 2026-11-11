# ModelSpec: clean version 1 design

Entry/output selection and construction were subsequently revised in the
[entry/output plan](../plans/2026-11-08-modelspec-entry-output.md). The source/result
examples below record the earlier design.

Status: version 1 implemented; deployment and scientific validation limits remain below.
Date: 2026-11-08.

The replacement starts at `schema_version: 1`. There is no old-schema migration,
compatibility interpreter, stage shim, or requirement to preserve the old
ModelSpec feature set. The user explicitly rejected mapping/index references
and the extra named-flow registry. ModelSpec has no registry or callable factory; both are absent from the study.

## Caller and document

```python
spec = ModelSpec.load("model_spec.yaml")

prediction = spec.run(source="from_raster", result="classes", raster=dataset)
path = spec.run(
    source="from_area",
    result="saved",
    geometry=polygon,
    time="2025-06",
    output="map.tif",
)
```

```yaml
schema_version: 1

inputs:
  raster: {}
  geometry: {type: object}
  time: {type: string}
  batch_size: {type: integer, default: 8}
  output: {type: string}

from_raster:
  call: project.accept_raster
  kwargs: {raster: !ref raster}

from_area:
  call: project.acquire_area
  kwargs: {geometry: !ref geometry, time: !ref time}

checked:
  call: project.validate_raster
  kwargs: {raster: !ref source}

prepared:
  call: project.prepare
  kwargs: {raster: !ref checked}

scores:
  call: project.predict_tiles
  kwargs: {raster: !ref prepared, batch_size: !ref batch_size}

classes:
  call: project.classify
  kwargs: {scores: !ref scores}

saved:
  call: project.write_raster
  kwargs: {raster: !ref classes, path: !ref output}
```

`project.*` denotes installed Python functions, not existing new GeoSave APIs.
The local Prefect smoke uses [operations.py](../studies/2026-11-08-modelspec/operations.py).

Only `schema_version` and `inputs` are document metadata. Other root names
identify calls. Callers select source/result directly. A deployment is ordinary
Python outside ModelSpec; the document does not define deployment names, named
execution presets, generated functions, or a fixed processing lifecycle.

## References: names and attributes only

| Reference | Meaning |
| --- | --- |
| `!ref raster` | The external argument named `raster` |
| `!ref prepared` | The result of the named computation in this invocation |
| `!ref source` | The result of the selected source node |
| `!ref raster.gs.geobox` | An existing object's attributes |
| `call: !ref model.predict` | An existing object's callable attribute |

No mapping-key syntax, integer indexing, expressions, slices, or calls inside
references. A dotted path uses Python attribute access; it does not guess keys.
A simple split-and-validate parser replaces AST parsing. Attribute references
remain because native accessors and bound methods are already concrete uses.

Functions can receive whole mappings/tuples and handle them in Python. Do not
introduce generic item-selection helpers or wrapper classes merely to emulate
removed reference syntax. There is no demonstrated need for that capability in
this candidate. Add a feature only when a real model recipe requires it.

Inputs and node results cannot share a name. `source` is reserved for selected
source output. Input names cannot collide with the `source`/`result` invocation
selectors. Unknown references are errors, never implicitly declared arguments.

## Execution contract

1. The caller selects a source and result(s). No declaration-order fallback.
2. Follow result dependencies, replacing `!ref source` with the selected source.
   Detect cycles for that selection before execution. Loading has no predefined
   selections to validate; planning checks each requested source/result pair.
3. Collect only inputs referenced on the selected path. Returning a native
   prediction does not require the destination used only by the writer.
4. Validate supplied inputs, fill declared defaults, and reject unknown/unused
   arguments before source execution. Copy mutable defaults per invocation.
5. During trusted execution preparation, import selected callables and check
   argument binding where Python exposes a signature. Loading/planning must not
   import or execute declared callables or resolve runtime properties.
6. Execute each selected node once per invocation. A fresh local mapping retains
   native results. A reference does not copy, compute, or serialize its value.
7. Return a native value, or a plain mapping when multiple results are requested.

```python
prepared = spec.run(source="from_raster", result="prepared", raster=dataset)
outputs = spec.run(source="from_raster", result=["scores", "classes"], raster=dataset)
```

An intermediate result request executes its dependencies. It does not override
those dependencies with caller-provided intermediate values. Separate requests
recompute; multi-result requests share work within one invocation.

Fresh execution state does not make shared rasters or models immutable or
thread-safe. Python functions own mutation, loops, branching, batches, per-sample
state, resource lifetime, and cleanup. The prototype retains all intermediates
until return; realistic memory behavior remains to be designed and measured.

## Sources and outputs

Alternative sources must satisfy the shared preparation contract. They need
compatible variables, units, grids and temporal coverage, not identical values.
Validation is an ordinary built-in or custom callable, never a privileged root
schema block. Reuse existing accessor validation rather than recreating it.

A selected source may return a whole native mapping for multiple modalities;
Python functions consume that mapping directly. Independent selectable source
groups are not part of this version. Geometry/time acquisition must define an
actual grid, and centroid dimensions need explicit units. The study simulates
acquisition; it does not validate those scientific choices.

Writers are ordinary nodes. They execute only if requested or required by a
requested result. Publishing may depend on saving first. Side-effect order must
follow dependencies or be implemented in one Python function, never inferred
from YAML position. Runtime failures do not imply rollback.

## Inputs and Prefect

Invocation inputs allow `batch_size` to go directly to prediction and `output`
to writing. Neither is forwarded through every source function.

The experimental input declarations support primitive `type` and `default`
fields. Empty declarations allow native local values. This type/default syntax
still needs review; do not expand it into a rich custom type language. Deployment
inputs should use paths or JSON requests rather than live raster objects.

ModelSpec exposes `load`, `save`, `plan`, and `run`. It has no `bind` or generated-callable interface.

Prefect integration is an ordinary project-owned Python function:

```python
from prefect import flow

@flow(persist_result=False)
def predict_file(path: str, output: str, batch_size: int = 8) -> str:
    spec = ModelSpec.load("model_spec.yaml")
    return spec.run(
        source="from_path",
        result="saved",
        path=path,
        output=output,
        batch_size=batch_size,
    )
```

The function supplies Prefect's public parameter signature. Source/result
selection is visible at the call site. There is no separate ModelSpec flow
object, registry, binding step, or signature-generation mechanism. Prefect's
own flow remains because Prefect is the requested orchestrator.

The study runs this outer function with result persistence off. Automatic
per-node tasks would change resource, memory and retry semantics and are not
included. Worker deployment must install the required Python code and recipe.
Actual remote worker deployment remains unverified.

## Trust and clean implementation

YAML parsing is inert, but executing configured Python functions is trusted code
execution. Attribute lookup can execute properties. The reduced reference
syntax is not a sandbox. No automatic fetching/installing custom code.

Production layout:

```text
model/spec/model.py       document, plan, execute, load/save
model/spec/reference.py   small inert name/attribute references
model/spec/__init__.py    intentional minimal exports
model/processing.py      useful independent numerical operations
project prediction.py    ordinary Prefect entrypoint calling ModelSpec.run
```

Do not copy the study wholesale. Implement the chosen contract cleanly; retain
independent ecosystem operations, not the old ModelSpec classes beneath a new
interface. See the [deletion/rebuild plan](../plans/2026-11-08-model-workflow.md).

## Outstanding checks

Large-scene resource lifetime; custom-code release packaging; real worker
execution; task-specific model processing and outputs. No old recipe conversion
or automatic migration is required. Useful numerical protections can be
re-expressed as new tests without preserving the old interface.

Evidence: [study README](../studies/2026-11-08-modelspec/README.md).
