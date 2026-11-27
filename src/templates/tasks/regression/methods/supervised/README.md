# Supervised dense regression

Predict one or more continuous variables per pixel. Each Dataset sample contains named `inputs` and floating-point `target` values with shape `(variable, y, x)`. Unknown target values use NaN.

Implement the split reader in `training/datasets.py`. Set the variable names, data selection and model in `configs/supervised.yaml`. The LightningModule scores finite targets in training, validation and test.

Run training after connecting the Datasets.

```bash
python main.py fit --config configs/supervised.yaml
```

`pipeline.py` connects the stages in `processing/`. The Dataset can call the same `processing.preprocessing.preprocess` on stored source rasters before producing its tensors. The DataModule augments prepared batches during training.

The region pipeline reads Sentinel-2 reflectance. Spatial prediction, the output variables and their storage format are project extension points.
