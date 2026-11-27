# Masked image modeling

Learn to reconstruct masked regions of the selected image input. Each Dataset sample contains named tensors under `inputs`. The training module creates the reconstruction target from the image.

Implement the split reader in `training/datasets.py`. Set the reconstruction input, patch size and mask ratio in `configs/mim.yaml`. Training and validation score the hidden patches.

Run training after connecting the Datasets.

```bash
python main.py fit --config configs/mim.yaml
```

`pipeline.py` connects the stages in `processing/`. The Dataset can call the same `processing.preprocessing.preprocess` on stored source rasters before producing its tensors. The DataModule augments prepared batches during training.

The region pipeline reads Sentinel-2 reflectance. Regional reconstruction, its interpretation and its storage format are project extension points.
