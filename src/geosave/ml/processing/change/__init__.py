"""Run a model over two acquisitions of one place: before and after rasters in, a change product out.

    before, after ──preprocessing──▶ sample ──prediction (model)──▶ output
        ──postprocessing──▶ product

Only preprocessing differs from a raster pipeline: it takes both
acquisitions and returns one sample tree, so the raster predictor and
postprocessor roles, and their implementations, serve change detection too.
"""
