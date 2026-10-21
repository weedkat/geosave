# Random training crops and complete-region evaluation

Train on whole, aligned Dynamic World tiles and crop paired image/label batches
with Kornia after device transfer. Keep deterministic geospatial chips available
for larger inference regions. Score every validation/test tile after overlapping
window scores have been recombined and padding removed.

1. Share local-grid construction on GeoSample between chip planning and full
   sample reads with resolution. Preserve lazy alignment before temporal assembly.
2. Keep DynamicWorld limited to complete image/label frames. Inference opens
   the saved catalog, plans chips explicitly, and batches reads through Chips.
   Apply paired crop/flips in the DataModule's training-only device hook,
   without padding or label remapping; training tiles must be at least the crop size.
3. Give Segmenter a tensor-window prediction method using native Kornia patch
   extraction/recombination. Validation/test update global metrics with restored
   full-tile logits and original targets, not individual window scores.
4. Update notebook prose, explicit inference chip configuration, and offline
   example tests. Verify paired masks, padding/ignore labels, full-region coverage
   across model window sizes, and a short Lightning fit/validate/test run.

The training example expects uniformly sized aligned tiles for normal batching;
large heterogeneous scenes still need bounded source-window sampling.
