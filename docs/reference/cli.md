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
geosave create demo --task segmentation --method supervised
```

| Input | Default | Meaning |
| --- | --- | --- |
| `NAME` | Interactive prompt | Destination directory. |
| `--task`, `-t` | Interactive selection | `segmentation`, `regression`, `classification`, `pretraining`, `lightning`, or `blank`. |
| `--method`, `-m` | Interactive selection | A method supported by the selected task. No method for `blank`. |
| `--force` | Off | Permit a non-empty destination while preserving existing startup files. |

The output is an editable project with local training implementations, config,
and method-specific scripts. It does not train or download data. Task/method
validation happens before writing files. See [available methods](training.md).

## Add a task and method

```bash
cd demo
geosave add-task --task segmentation --method fixmatch
```

`--task` / `-t` and `--method` / `-m` have the same selectors as creation,
without the blank option. Existing methods are disabled in the selector;
explicitly selecting an existing method reports an error without changing files.
Run inside a workspace containing `main.py`. Each method gets separate local
Python, config, processing-document, and script paths. Shared startup files are
not copied again. Existing incidental files trigger overwrite/skip prompts.

## Publish a model

`geosave publish [CHECKPOINT] --repo ORG/NAME` is a placeholder: it prints that
it is not implemented yet and exits with status 1. Release a checkpoint with
`scripts/segmentation/supervised/publish.py` from the segmentation starter in the meantime.

## Acquire model rasters

Install `geosave[flows]` to use `geosave-flows ingest`.

| Input | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--anchor JSON` | Yes | — | Validated `coordinates`, `geojson`, or `raster` [anchor configuration](workflow.md). |
| `--output PATH` | Yes | — | New folder for the rasters and their catalog. |
| `--spec PATH` | Yes | — | Model spec with a STAC recipe for every declared raster. |

Use local `--output` and `--spec` paths for now: the CLI currently rewrites
remote URL schemes through `pathlib.Path`. The Python flow accepts location
strings directly.

The command waits for loading and writing, then prints the path of the written
`catalog.parquet`. It needs access to the configured endpoints/assets. It
does not run preprocessing or training. See [ingestion](../guides/workflows.md)
for grid, time, write, and failure behavior.

## Shared options

`--help` displays help and exits on every command. At the top level,
`--show-completion` prints shell completion code and `--install-completion`
installs it for the current shell. Both CLIs currently call automatic `.env` discovery, which can start from
the installed library rather than your workspace. Export required settings
explicitly until workspace lookup is corrected; generated training startup
does not itself load `.env`.
