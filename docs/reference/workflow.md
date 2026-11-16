# Prediction flow

`geosave_flows` runs a released pipeline as a Prefect flow, installed with
`pip install "geosave[flows]"`. Every argument is JSON, so the flow deploys
and schedules as it is called.

```python title="Adaptation example — use your release and region"
from geosave_flows.predict import predict_region

stores = predict_region(
    repo="org/dynamic-world",
    region={"kind": "coordinates", "latitude": -6.2, "longitude": 106.8,
            "shape": 1024, "resolution": 10, "timespan": "2024-07"},
    root="output/land-cover",
)
```

A run records one `ingest` task, retried on failure, and one
`predict_to_zarr` task per acquisition.

::: geosave_flows.predict.predict_region

## Region configuration

::: geosave_flows.configs
