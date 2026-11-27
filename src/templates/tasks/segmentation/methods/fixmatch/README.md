# FixMatch semantic segmentation

This starter combines labeled image/mask chips with unlabeled imagery.
Weakly augmented unlabeled pixels provide confident pseudo-labels for
the corresponding strongly augmented pixels.

1. Point `training.datasets.RasterDataset` in `configs/fixmatch.yaml` at aligned
   image and label GeoTIFFs. The default expects Sentinel-2 B02, B03, B04 and
   B08, two classes, and ignore index 255.
2. Implement `training/unlabeled.py` for your imagery. Return a sample containing
   `inputs` with named, prepared tensors. Match the labeled band's order, units
   and grid context. Only photometric transforms belong in the strong-view stage.
3. Run the command from this directory.

```bash
python main.py fit --config configs/fixmatch.yaml
```

`training/fixmatch_module.py` adds the pseudo-label objective to the supervised
training logic in `training/module.py`. `training/fixmatch_datamodule.py` combines
the two training loaders and constructs the aligned weak and strong views.
Changing your model or a training parameter requires editing one experiment config.

The default run saves the last checkpoint and skips validation. Implement the
validation and test policies before enabling them. `pipeline.py` connects the
prediction stages in `processing/`. Implement regional source selection in
`processing/ingestor.py` before calling that pipeline on a region.
