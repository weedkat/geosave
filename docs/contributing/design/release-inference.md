# Release and inference

Updated against library source on 2026-11-15. Processors are implemented and
stored in releases; this page records that contract and what remains open.
See [samples and chips](samples-chips.md) for chip placement and merging, and
the user guide [Release and run a model](../../guides/release.md) for a runnable
example.

## Current contract

Two registries in `model/processing.py`:

```text
sub-processors   register_processing(role, name)   one Protocol per role
processors       register_processor(name)          hold the model and its sub-processors
```

| Role | Methods | Owns |
| --- | --- | --- |
| ingestor | `ingest(region) -> list[Dataset]` | acquiring the rasters of a region, for inference only |
| preprocessor | `preprocess(raster) -> Dataset`, `normalize(image) -> Tensor` | band selection and decoding (lazy, whole raster) and batch standardization; shared by training and inference |
| postprocessor | `postprocess(logits: DataArray) -> Dataset` | decoding merged `(band, y, x)` output on its grid |
| processor | `predict_region`, `predict_path`, `predict_raster` | running the model over what the caller has |

`model/dense_chips.py` holds the one library processor, `dense_chips`:

```text
raster ─preprocess─▶ image ─Chipper─▶ chips ─normalize─▶ model(image=batch) ─ChipMerger─▶ logits ─postprocess─▶ product
```

`build_processing(role, spec)` and `build_processor(spec, model)` construct
from `BuildSpec`; an argument named after a role is itself a nested
`{name, init_args}`. Each built object's `spec` records its selector and every
constructor argument, defaults included.

```python title="What a release stores and rebuilds"
processor = build_processor(spec, chain)
GeoSaveModel.from_chain(chain, processor).save_pretrained("release")
# config.json: stages=[...], processor={"name": "dense_chips", "init_args": {
#     "ingestor": {...}, "preprocessor": {...}, "postprocessor": {...},
#     "size": 256, "stride": 192, "mode": "reflect", "window": "hann", "batch_size": 8}}
processor = load_processor("release")   # model in eval mode, processor rebuilt around it
```

`ARTIFACT_FORMAT_VERSION` is 2; loading rejects other versions and a different
GeoSave version. A release without a processor still loads as a model;
`load_processor` raises `ValueError` for it. The library registers no
ingestor, preprocessor or postprocessor; projects do, and must import the
registering module before `load_processor`. `workspace/dynamicworld`
(`modules/processing.py`, `scripts/publish.py`, `scripts/predict_region.py`)
runs the full path: train, release, predict a region.

The ModelSpec declaration, its graph executor and the plain-function registry
sketches are superseded by these classes.

## Decoding by task

| Kind | Chip output | Merged, then | Published as |
| --- | --- | --- | --- |
| segmentation | logits | argmax; optional per-class thresholds from calibration | uint8, `CFFlags` from classes/colors, nodata 255 |
| regression | values | — | float32, units, NaN nodata |
| detection | boxes, scores, classes | NMS across chip overlaps | GeoDataFrame |
| classification | logits | — | GeoDataFrame per chip footprint |

Postprocessors implement these decodings; `dense_chips` covers the two dense
rows. Merging logits before argmax is what makes overlap blending meaningful.
`softmax_argmax` still raises, so the calibration callback depending on it is
unavailable.

## Open

| Question | Notes |
| --- | --- |
| Named or stacked inputs | `dense_chips` calls `model(image=batch)` only. Prithvi-TL and Clay also need time/location context; S1 + S2 early fusion wants one tensor |
| Model context | derive `temporal_coords`/`latlon` from the chip's `transform`, `crs`, `time`, or keep `model_context(raster)` per encoder |
| Normalization statistics | measured on the prepared data and passed as preprocessor arguments, or typed by hand |
| Missing steps or sources | zero-fill with a validity input, or drop the sample; must match between training and inference |
| How a time step is filled | first, least cloudy, composite; see [sample rows](samples-chips.md#sample-rows) |
| Offline inference | preload ingest, preprocessing and chips into a store, then predict from it; may change the processor's method suite |
| Implementation availability | a release names classes, it does not ship code; a project's registry module must be importable where it loads |
| Whole-scene evaluation | training uses random crops; validation could run `predict_raster` and score merged scenes |
| Vector outputs | detection, instances and chip classification need their own processors and merge rules |
| Memory | `ChipMerger` holds full output accumulators; very large rasters need streaming writes |
