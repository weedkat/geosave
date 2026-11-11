# Run recipes with Prefect

```python
from prefect import flow
from geosave.model.spec import ModelSpec

@flow(persist_result=False)
def predict_file(path: str, output: str, batch_size: int = 8) -> str:
    spec = ModelSpec.load("model_spec.yaml")
    return spec.run(
        entry="from_path", output="saved",
        path=path, destination=output, batch_size=batch_size,
    )
```

This is a project-owned entrypoint: its recipe must declare those inputs and
calls. Install `geosave[flows]`, your model dependencies, and your project's
Python functions in the worker environment. The explicit signature supplies
Prefect's public inputs; ModelSpec executes the selected dependency graph.

Use paths or JSON requests at deployment boundaries. An acquisition function can
accept geometry/time or a centroid and explicit dimensions, then return the
native raster expected by preparation. Implement search and loading using
[STAC APIs](stac.md), with explicit grid and coverage decisions.

ModelSpec does not register each call as a Prefect task or create deployments.
Functions own batching, resources and advanced control flow. Output functions
execute only when selected or needed by another selected result. Retry policy
must account for side effects; a failed recipe does not roll them back.

The previous fixed ingestion flow and `geosave-flows` CLI were removed with the
old ModelSpec engine. Independent region configurations and label tasks remain
available in [the workflow reference](../reference/workflow.md).
