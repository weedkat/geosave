# Supervised chip classification

Predict one class per image chip. Supply prepared tensors under `inputs` and a scalar integer `target` for each chip.

Implement the split reader in `training/datasets.py` and set the class count and data selection in `configs/supervised.yaml`. The LightningModule trains and evaluates the supplied labeled chips.

Run training after connecting the Datasets.

```bash
python main.py fit --config configs/supervised.yaml
```

`pipeline.py` connects the stages in `processing/`. The Dataset can call the same `processing.preprocessing.preprocess` on stored source rasters before producing its tensors. The DataModule augments prepared batches during training.

The region pipeline reads Sentinel-2 reflectance. Prediction over chip footprints, class-label postprocessing and writing are project extension points.
