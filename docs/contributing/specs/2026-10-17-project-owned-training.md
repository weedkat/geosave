# Project-owned training recipes

## Intent

Generated projects own the LightningModule, LightningDataModule, Dataset, and
sample definitions that describe their training method. Users can edit training,
augmentation, batching, and evaluation together without subclassing a library
recipe. GeoSave continues to provide reusable geodata, model construction,
metrics, losses, transforms, builders, and release operations.

This change does not redesign ModelSpec, preprocessing, postprocessing, the
release format, or remote-code execution.

ModelSpec is flagged pending a decision on publishing and executing preprocessing
and postprocessing code. Its current implementation remains available, but this
migration does not commit to its YAML structure or expand its processing language.

## Workspace creation

Restore the two-stage terminal selector previously implemented in commit
`35d8e26d`: choose a task, then choose one of that task's methods. Use Typer
options `--task` / `-t` and `--method` / `-m` for explicit selection:

```bash
geosave create landcover --task segmentation --method supervised
geosave create landcover --task segmentation --method fixmatch
geosave create experiment --task blank
```

Omitted task or method options prompt for the missing selection. Blank creates
shared startup only and has no method selection; reject an explicit method for
blank. Validate task/method combinations before creating files. Keep existing
non-empty directory protection and `--force` behavior. Restore the selection
experience without restoring obsolete project metadata or old implementation
bugs.

Offer segmentation with supervised/FixMatch, regression with supervised/Mean
Teacher, classification with supervised, pretraining with experimental MIM,
and Lightning with a boilerplate method. Show descriptions in the interactive
selector, including the experimental status of MIM. Retain `add-task` for adding
another method later, using the same task/method selection rules.
Remove the overlapping `segmentation`, `regression`, and `custom_lightning`
flat template directories rather than maintaining aliases. Organize bundled
recipes by task and method; their generated config names can remain descriptive,
such as `segmentation-supervised.yaml`.

## Generated code

Keep distributable template resources under `src/templates/`.
Generated training implementations extend Lightning directly, or another local
recipe where the existing method shares supervised behavior. They do not import
`geosave.ml.tasks`.

Use method-specific Python packages, for example:

```text
modules/
  segmentation/
    supervised/
      module.py
      data.py
      datasets.py
    fixmatch/
      module.py
      data.py
      datasets.py
configs/
  segmentation-supervised.yaml
  segmentation-fixmatch.yaml
```

Each recipe owns its sample definitions. A Dataset remains an ordinary PyTorch
Dataset. Users can point multiple configurations at the same project Dataset
when its returned samples satisfy their methods. Adding a second recipe must
not overwrite the first recipe's Python implementation. Existing shared-file
collision prompts remain in effect.

Retain GeosaveCLI and reusable helpers where they simplify generated callers.
Keep default training behavior unchanged while relocating its ownership.
Preserve the offline runnable Lightning boilerplate; geospatial Dataset stubs
must clearly state that real data and implementation are required.

## Publication

Update the starter publication script to load its local LightningModule and
export the trained native model through the existing release API. Workspace
upload patterns must include the generated local packages. Runtime deployment
must not import uploaded training code.

Contributing a new deployment architecture remains separate from changing a
training method. Existing registered architectures can publish new weights
without a new library implementation.

## Migration

Remove library-owned training recipe implementations after moving their callers,
documentation, and behavioral tests to generated projects. Do not retain aliases
or parallel library execution paths. Trace callbacks and method inheritance
before moving or removing them; preserve independently useful callbacks and
document the contracts they expect.

Do not edit notebooks in this change. Report notebook imports of removed task
paths as an explicit migration limitation, if present.

## Verification

- Exercise interactive task/method selection, explicit options, partially
  supplied selections, blank creation without a method prompt, invalid
  combinations, and non-empty directory protection through the CLI.
- Generate each recipe and parse its configuration against its local classes.
- Run focused offline Lightning training and validation for generated recipes,
  preserving meaningful existing tests for target masks, augmentation, method
  behavior, and metrics.
- Save and restore a generated local Module checkpoint, export its trained model,
  and compare released inference outputs with the trained model.
- Add two methods to one workspace and verify their local implementations remain
  independently importable.
- Inspect the built wheel for template resources and run focused lint checks.

## Breaking changes and open review

Removing `geosave.ml.tasks` recipes breaks their import paths. Template aliases
are removed. Creation without `--task` becomes interactive; automation should
pass both task and method, or `--task blank`, explicitly. ModelSpec behavior and
deployment restrictions are unchanged; ModelSpec's future is deferred until
processing-code publication and execution are settled. Review the task/method
choices and generated package layout before writing the implementation plan.
