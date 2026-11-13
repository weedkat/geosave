# Model input contracts

ModelSpec is disabled pending redesign. Prepare inputs in ordinary project
functions, using native GeoSave/xarray operations and explicit tensor conversion.
See [prepare model inputs](../guides/model-inputs.md) for a working function
and [ModelSpec status](../guides/model-spec.md) for the paused recipe interface.

Each source must satisfy its consumers: variable meaning and order, units,
temporal selection, missing values and grid placement. Raster validation checks
geometry; project functions check scientific suitability. Reprojection,
resampling, observation selection and materialization remain explicit.

Derive model context after selecting the actual crop and observation.
Geometry-changing augmentation requires an explicit context policy; tensor
augmentation does not update geographic metadata. Project functions own model
state, batching, resources and output interpretation.
