# Segmentation / supervised

Edit `modules/segmentation/supervised/module.py`, `data.py`, and `datasets.py`
to develop the model, training method, augmentation, batching, and evaluation.
These are project-owned implementations, not subclasses of library training tasks.
GeoSave supplies reusable model, geodata, metric, loss, and transform helpers.

`RasterDataset` reads the explicit `images`/`targets` pairs in the config. Replace
the example paths with your files; each pair is an already selected observation
or chip. No catalog is required. The recipe selects B02/B03/B04/B08, decodes packed
reflectance, and returns `(4, height, width)`; labels must be one `(y, x)` class band
on the prepared image's exact grid. Set `num_classes` and `ignore_index` consistently
in the Dataset, DataModule, model head, and loss.

Label nodata becomes `ignore_index`. Fractional/out-of-range labels and nonfinite
image pixels fail explicitly. If imagery has gaps, declare a scientifically
appropriate fill/mask policy in the recipe and adapt the Dataset's label policy;
ignoring a label alone does not prevent NaNs spreading through a convolution.
This example does not infer observations, align grids, chip full scenes, or fill
missing bands. Adapt the project Dataset for those needs.
Keep training and validation scenes or geographic regions separate.
The default encoder downloads pretrained weights; review that setting before use.

```bash
python main.py fit --config configs/segmentation-supervised.yaml
```

The version 1 ModelSpec is at `configs/segmentation-supervised/model_spec.yaml`.
The Dataset returns tensors before normalization. The DataModule augments training
batches and calls `spec.run(entry="from_tensor", output="normalized", value=tensor)`
in every split. Inference uses entry `encoded` with `raster=scene` and the same
`normalized` output. Save the recipe and matching project code with the checkpoint.
Use `geosave.model.processing.to_raster` to place per-chip logits on their image
grid. Merge overlapping logits before selecting classes. These operations belong
to project Python code. The default input contract rejects a retained time axis.

Add another task/method with `geosave add-task --task TASK --method METHOD`.
Each method has separate Python, config, and script paths. To share a Dataset,
point both configs to its import path when their sample contracts match.

Publish registered model stages with `python scripts/segmentation/supervised/publish.py ORG/MODEL CHECKPOINT`.
