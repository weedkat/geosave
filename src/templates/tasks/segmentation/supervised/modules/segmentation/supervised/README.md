# Segmentation / supervised

Edit `modules/segmentation/supervised/module.py`, `data.py`, and `datasets.py`
to develop the model, training method, augmentation, batching, and evaluation.
These are project-owned implementations, not subclasses of library training tasks.
GeoSave supplies reusable model, geodata, metric, loss, and transform helpers.

Implement the Dataset stubs before fitting. Match the sample definitions in
`data.py`.
Keep training and validation scenes or geographic regions separate.
The default encoder downloads pretrained weights; review that setting before use.

```bash
python main.py fit --config configs/segmentation-supervised.yaml
```

The existing ModelSpec is at `configs/segmentation-supervised/model_spec.yaml`.
Its future is on hold pending preprocessing/postprocessing code publication design.
This migration preserves its current behavior. Required inference transforms
must match training; random training augmentation is not inference preparation.

Add another task/method with `geosave add-task --task TASK --method METHOD`.
Each method has separate Python, config, and script paths. To share a Dataset,
point both configs to its import path when their sample contracts match.

Publish registered model stages with `python scripts/segmentation/supervised/publish.py ORG/MODEL CHECKPOINT`.
