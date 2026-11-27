# Create a workspace

`geosave create` copies an editable starter into a directory. For an
interactive selection, run:

```bash
geosave create landcover
```

For the same operation in a script, supply the task and method:

```bash
geosave create landcover --task segmentation --method supervised
```

| Input | Behavior |
| --- | --- |
| `NAME` | Destination directory. Prompted when omitted. |
| `--task`, `-t` | Select a task, or `blank` for only the common startup files. |
| `--method`, `-m` | Select a method belonging to the task. A task with a single method selects it automatically. |
| `--force` | Permit creation in a nonempty directory. Unrelated files are preserved. A conflicting destination path stops creation before copying. |

The [recipe catalog](training.md) lists the available task and method
pairs. The selections identify the files to copy. You can organize additional
methods, Datasets and experiments however you like after creation.

The starter's README describes its expected inputs and supported operations.
Training runs through `python main.py fit --config configs/<method>.yaml`.
Geospatial starters include an editable `pipeline.py` and `processing/`
package. Their README identifies any unfinished stages.

See the [training walkthrough](../advanced/training.md) for an offline run and
[model releases](../advanced/release.md) for exporting a trained chain.
