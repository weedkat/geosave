# Supervised semantic segmentation

Train on aligned image and class-mask chips. The starter expects single-observation
GeoTIFF images with bands B02, B03, B04 and B08 in Sentinel-2 reflectance units
or CF-packed reflectance. Each corresponding label file has one integer class
band on exactly the same grid. The example has two classes and uses 255 for
ignored pixels.

## Train

1. Set the paired image and label paths in `configs/supervised.yaml`. Each pair
   should cover the chip size expected by the model.
2. Adjust the model, number of classes, ignore index and optimizer in that config.
3. Run the command from the project directory.

```bash
python main.py fit --config configs/supervised.yaml
```

The default DINOv3 encoder needs its pretrained weights. For an offline CPU
experiment choose the registered `small_cnn` encoder and `unet` decoder.
Training saves `last.ckpt` under `artifacts/<model_name>/version_N/checkpoints/`.
To continue a run, use its resolved config and pass `--ckpt_path PATH`.

## Where the code lives

- `training/datasets.py` reads paired rasters and returns
  `{"inputs": {"image": pixels, **context}, "target": target}`. List the
  model's context names in the Dataset config. Set `grid: true` if the model
  also takes the source transform and CRS. Image preparation is shared with
  inference in `processing/preprocessing.py`.
- `training/datamodule.py` batches chips and applies joint image/mask training
  augmentations. `training/module.py` owns the model, loss and metrics.
- Validation and test hooks raise `NotImplementedError` pending a project
  evaluation policy. The starter disables validation and saves the last
  checkpoint without monitoring validation metrics.
- `pipeline.py` connects ingestion, preprocessing, scene prediction,
  postprocessing and persistence. Implement source and observation selection in
  `processing/ingestor.py` before calling it with a region. For stored images,
  reuse the preprocessing and predictor directly. The predictor expects a model in
  evaluation mode that returns logits on the input grid. The output COG carries
  integer class IDs and uses −1 where logits are missing.
- `python scripts/prepare_example.py` demonstrates decoding one packed image.
  `python scripts/publish.py --help` describes model-chain release and archiving
  the training run. The editable pipeline belongs with the project code.
