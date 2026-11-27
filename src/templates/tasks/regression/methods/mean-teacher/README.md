# Mean Teacher dense regression

Train a regression model on labeled pixel values and unlabeled imagery. The student learns from known target values and matches the moving-average teacher on unlabeled images.

Implement the labeled reader in `training/datasets.py` and the unlabeled reader in `training/unlabeled.py`. Set the variable names and data selection in `configs/mean-teacher.yaml`. Edit `training/teacher_module.py` for the consistency loss and teacher updates, and `training/teacher_datamodule.py` for the weak and strong views.

Run training after connecting the Datasets.

```bash
python main.py fit --config configs/mean-teacher.yaml
```

`pipeline.py` connects the stages in `processing/`. The Dataset can call the same `processing.preprocessing.preprocess` on stored source rasters before producing its tensors. The DataModule augments prepared batches during training.

The region pipeline reads Sentinel-2 reflectance. Spatial prediction, the output variables and their storage format are project extension points.
