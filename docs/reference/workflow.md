# Workflow inputs, flows, and tasks

Prefect flows run complete jobs from validated inputs.
See [Ingest data](../guides/workflows.md). `prepare_dense_data` is a declared
placeholder that raises NotImplementedError; use ModelSpec.prepare for preparation.

## Anchor configuration

::: geosave_engine.workflow.configs

## Flows

::: geosave_engine.workflow.flows

## Write a sample

::: geosave_engine.workflow.tasks.sample
    options:
      members:
        - write_sample
        - SampleFormat

## Label files

::: geosave_engine.workflow.tasks.labels
    options:
      members:
        - find_labels
        - read_labels
