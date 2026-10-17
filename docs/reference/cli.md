# Command-line inputs and outputs

```bash
geosave --help
geosave create --help
geosave add-task --help
geosave publish --help
geosave-flows ingest --help
```

These commands show the installed version's accepted inputs. Use the
[workspace guide](../guides/workspaces.md) or
[ingestion guide](../guides/workflows.md) for a complete activity.

## Create a project

```bash
geosave create demo
```

| Input | Default | Meaning |
| --- | --- | --- |
| `NAME` | Interactive prompt | Destination directory, relative to the current directory unless absolute. |
| `--force` | Off | Create inside a directory that is not empty. |

The output is an editable directory with shared startup files and no task
configs. It does not start training or download data. A non-empty target is
refused without `--force`.

## Add a task

```bash
cd demo
geosave add-task segmentation-supervised
```

| Input | Default | Meaning |
| --- | --- | --- |
| `TASK` | Interactive selection | `segmentation-supervised`, `regression-supervised`, or `lightning-boilerplate`. |

The output is `configs/<task>.yaml` plus the template's shared files
(`configs/model_spec.yaml`, `modules/datasets.py`, scripts). Run it inside a
workspace, which is a directory containing `main.py`. Existing files trigger
overwrite/skip prompts, so a second task keeps your edited shared files.

## Publish a model

`geosave publish [CHECKPOINT] --repo ORG/NAME` is a placeholder: it prints that
it is not implemented yet and exits with status 1. Release a checkpoint with
`scripts/publish.py` from the segmentation starter in the meantime.

## Acquire model rasters

Install `geosave[flows]` to use `geosave-flows ingest`.

| Input | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--anchor JSON` | Yes | — | Validated `coordinates`, `geojson`, or `raster` [anchor configuration](workflow.md). |
| `--output PATH` | Yes | — | New local sample directory. |
| `--spec PATH` | Yes | — | Model spec with a STAC recipe for every declared raster. |
| `--format` | No | `zarr` | `zarr` or `geotiff`, applied to each raster. |
| `--write-options JSON` | No | `{}` | Native writer options; publication controls `compute`, `layout`, and `overwrite` are rejected. |

The command waits for acquisition and writing, then prints the completed
sample directory. It needs access to the configured endpoints/assets. It
does not run preprocessing or training. See [ingestion](../guides/workflows.md)
for grid, time, write, and failure behavior.

## Shared options

`--help` displays help and exits on every command. At the top level,
`--show-completion` prints shell completion code and `--install-completion`
installs it for the current shell. Both CLIs load an available `.env` from
the current directory or a parent before running a command.
