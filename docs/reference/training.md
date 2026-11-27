# Training kits

Each starter supplies its own editable `training` package and a
`configs/<method>.yaml` file. Start with the project's README for the inputs
and a training command. Use `geosave create NAME --task TASK --method METHOD`
to select a kit. [Workspace walkthrough](../advanced/training.md)

| Task | Method | Supplied behavior |
| --- | --- | --- |
| `lightning` | `boilerplate` | Offline synthetic Dataset, training and validation. |
| `segmentation` | `supervised` | Reads aligned image/label rasters and trains on labeled chips. Validation and test raise `NotImplementedError` pending the evaluation input contract. |
| `segmentation` | `fixmatch` | Confidence-gated training with weak and strong views. Supply unlabeled samples. |
| `classification` | `supervised` | One class per sample. Implement the project Dataset reader. |
| `regression` | `supervised` | Continuous targets with missing values. Implement the project Dataset reader. |
| `regression` | `mean-teacher` | Labeled and unlabeled samples with an exponential-moving-average teacher. Implement the project Dataset readers. |
| `pretraining` | `mim` | Experimental masked image reconstruction. Implement the project Dataset reader. |

In geospatial starters, `pipeline.py` connects the project modules in
`processing/` for ingestion, preprocessing, prediction, postprocessing and
persistence. Unfinished stages raise `NotImplementedError` when called.
The synthetic Lightning starter only needs its training kit.

For the sample shapes and learning objectives, see [training contracts](../concepts/training.md).
Library-owned registries, metrics, transforms, losses and callbacks are
documented under [training tools](training-tools.md).
