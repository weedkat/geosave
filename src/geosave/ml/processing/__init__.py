"""Pipelines run a model over what a caller has; one package per input shape.

    raster/   RasterPipeline.predict(rasters)            one acquisition, any sensors
    change/   ChangePipeline.predict(before, after)      two acquisitions of one place
    prompt/   PromptPipeline.predict(rasters, prompts)   rasters and the shapes to look at

Within a pipeline every handoff has one type, stated by its roles'
signatures. A pipeline whose sample is a raster tree reuses the raster
predictor and postprocessor roles; only the steps that take another input
have roles of their own.
"""
