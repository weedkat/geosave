"""Run a model over the rasters of one acquisition: raw rasters in, the model's product out.

    region ──ingestion──▶ rasters ──preprocessing──▶ sample
        ──prediction (model)──▶ output ──postprocessing──▶ product

`RasterPipeline` orchestrates inference. Each step is a sub-processor in a
package holding its abstract role in `base.py`, its registry and the
library's implementations, named by the task they serve: `dense` covers
per-pixel segmentation and regression. The roles' signatures are the
contract between steps; the predictor's `Output` and the postprocessor's
`Product` are type parameters a task fills in. A project writes its own
sub-processor by subclassing a role, never a library implementation, and
moves it into the library once the experiment is done. Training uses the
preprocessor alone, so inference prepares its input exactly as training did.
"""
