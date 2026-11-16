"""Run a model over rasters: raw rasters in, the model's product out.

    region ──ingestion──▶ rasters ──preprocessing──▶ sample
        ──prediction (model)──▶ output ──postprocessing──▶ product

`RasterPipeline` orchestrates inference: it runs the steps around a model.
Each step is a sub-processor, called directly, in a package holding its
abstract role in `base.py`, its registry and the library's implementations,
named by the task they serve: `dense` covers per-pixel segmentation and
regression. A sub-processor's `__call__` declares with `@chain_step()` the
values it takes and returns by name, which is the contract a pipeline checks
between steps. A project writes its own sub-processor by subclassing a role,
never a library implementation. Training uses the preprocessor alone, so
inference prepares its input exactly as training did.
"""
