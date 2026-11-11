# ModelSpec feasibility evidence

The prototype has been replaced by `geosave.model.spec.ModelSpec`. Its raster,
reference, and execution checks now live in `tests/model/spec/test_model.py`.
`prefect_smoke.py` runs an explicit outer Prefect flow against the production
implementation. The adjacent operations use synthetic acquisition and a toy
model; they do not establish scientific readiness or real worker deployment.
