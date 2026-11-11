# Model input contracts

```python
spec = ModelSpec.load("configs/segmentation-supervised/model_spec.yaml")
image = spec.run(entry="encoded", output="normalized", raster=scene)
```

This generated recipe selects bands, decodes reflectance, converts to tensors,
and normalizes. These are ordinary named calls, not built-in ModelSpec stages.
See [ModelSpec recipes](../guides/model-spec.md) for the complete syntax.

Each source must satisfy its consumers: variable meaning and order, units,
temporal selection, missing values, and grid placement. Native raster validation
checks geometry; project functions check scientific suitability. Reprojection,
resampling, observation selection, and materialization remain explicit.

Pass native values directly. `!ref prepared` refers to a named result and dotted
paths read attributes; dictionary selection belongs in Python. A function can
assemble model keyword arguments from pixels and geospatial context together.
Derive context after selecting the actual crop and observation. Geometry-changing
augmentation requires an explicit context policy; tensor augmentation does not
update geographic metadata.

Functions own model state, loops, batches, resources, and output interpretation.
Multiple requested results share computation within one invocation. Separate
invocations retain no intermediate results and recompute their dependencies.
