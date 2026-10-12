# Workflow inputs, flows, and tasks

Prefect flows run complete jobs from validated inputs. They ship as
`geosave_flows`, installed with `pip install "geosave[flows]"`. See [Ingest data](../guides/workflows.md).

## Anchor configuration

::: geosave_flows.configs

## Flows

::: geosave_flows.ingest

## Write a sample

::: geosave_flows.tasks.sample
    options:
      members:
        - write_sample
        - SampleFormat

## Label files

::: geosave_flows.tasks.labels
    options:
      members:
        - find_labels
        - read_labels
