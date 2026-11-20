"""Run a promptable model over rasters and the shapes to look at, such as SAM over a scene.

    rasters ──preprocessing──▶ sample ──prediction (model, prompts)──▶ output
        ──postprocessing (prompts)──▶ product

Preprocessing is a raster preprocessor: prompts do not change how pixels
are prepared. Prompts, a GeoDataFrame of points or boxes in map
coordinates, travel beside the sample to the predictor, which places them
in pixels, and to the postprocessor, which keeps their attributes.
"""
