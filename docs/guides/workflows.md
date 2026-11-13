# Optional ingestion flows

Prefect orchestration belongs in `geosave_flows`, separately from `geosave`.
Compose explicit native geodata operations in project-owned flows.

ModelSpec is disabled pending redesign. The old recipe executor is unavailable;
see [ModelSpec status](model-spec.md). Directory label ingestion is also paused
while STAC publication is rebuilt; existing label tables remain readable.
