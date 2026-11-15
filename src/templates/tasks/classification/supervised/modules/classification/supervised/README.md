# Processing

`processing.Reflectance` is this method's preprocessor, a
`geosave.ml.processing.roles.Preprocessor`. Raster datasets call `preprocess`
on each raster and the DataModule calls `normalize` on each batch. Configure it
under `preprocessor:` by `class_path`, or by `name` once registered with
`PREPROCESSORS.register`.

Releases do not carry a processor yet: publishing one also needs an ingestor,
predictor and postprocessor for this method.
