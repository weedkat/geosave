# Execute a model processing recipe

```python
spec = ModelSpec.load("model_spec.yaml")
prepared = spec.preprocess({"optical": raster})
values = spec.encode(prepared)
inputs = spec.transform(spec.model_inputs(values))
prediction = model(**inputs)
outputs = spec.decode(prediction, values)
```

ModelSpec v3 keeps processing in YAML and executes installed Python operations.
Native xarray values remain available until an explicit tensor conversion reads
pixels. Every invocation returns plain mappings; the spec does not cache sample
state or execute the model.

## Declare preparation and encoding

```yaml
schema_version: 3
rasters:
  optical:
    variables: [red, nir]
    coordinates: [time, y, x]
    require_crs: true
preprocessing:
  reflectance:
    call: !ref optical.gs.mask_and_scale
encoding:
  image:
    call: geosave.model.processing.to_tensor
    kwargs:
      raster: !ref reflectance
      dims: [band, time, y, x]
      dtype: float32
inputs:
  image: !ref image
pixels: [image]
tensors:
  image:
    shape: [2, time, height, width]
    dtype: float32
transforms:
  image:
    call: geosave.model.processing.normalize
    kwargs:
      value: !ref image
      mean: [0.15, 0.27]
      std: [0.19, 0.17]
      dim: -4
```

The example model takes unbatched `(band, time, y, x)` inputs. Dataset variables
become bands in the order selected by `rasters.optical.variables`. The explicit
`dims` controls tensor conversion; it is not inferred from an encoder name.
The normalizer uses the band axis `-4`, which remains the band axis after a
DataLoader adds a leading batch dimension. A model expecting `(time, band, y, x)`
uses that dimension order, `[time, 2, height, width]`, and `dim: -3` instead.

Numeric shape entries constrain sizes. Symbolic entries must agree across all
constrained inputs in one call. `null` leaves an axis independent. Dtype and
shape are checked before and after `transform`; conversion belongs in encoding.
For a DataLoader batch, call `spec.transform(inputs, batched=True)`.

`pixels` explicitly names the inputs that generated training methods may augment
as images. It does not declare that arbitrary geometry changes are valid for a
model with spatial context. The current generated augmenters expect trailing
`(band, y, x)` axes; use that encoding layout for their image augmentation, or
provide a method-specific augmenter for a different layout.

## Keep the execution boundaries explicit

| Method | Scope and result |
| --- | --- |
| `preprocess(rasters)` | Validate/select a supplied sample and run native operations; return selected rasters plus named results |
| `encode(prepared)` | Explicitly construct tensors/context; retain supplied and newly computed values |
| `model_inputs(values)` | Bind only the model's named arguments; intermediate state stays outside the call |
| `transform(inputs, batched=False)` | Shared input operations after optional training augmentation; return the original argument names with updated values |
| `decode(prediction, values)` | Process one unbatched model output using that same sample's retained state |
| `postprocess(prediction)` | Process merged outputs, for example choosing classes after merging logits |

Loading validates references and stage order without importing call targets.
Execution invokes installed Python code and is not a sandbox. A model artifact
needs the recipe, fitted parameters/state, and compatible installed dependencies.
Save the resolved recipe with the training run; loading weights alone does not
recover an editable workspace's former recipe.

The sample must already have sufficient spatial and temporal support. The caller
chooses source observations, temporal windows, target grids, interpolation,
neighborhood halos, and chip layout. ModelSpec does not move calls across these
boundaries. For example, scene-level percentile normalization must run over the
scene, and a terrain derivative needs its neighborhood before the final crop.

## Share training and inference processing

A project Dataset returns `spec.model_inputs(spec.encode(prepared))`. The
DataLoader batches those arguments. The generated DataModule performs training
augmentation and then calls `spec.transform(inputs, batched=True)` in every split.
Inference calls the same transform method. There is no separate inference
normalizer and no model-to-training import dependency.

Encoding returns state associated with exactly one sample. If cropping or frame
selection changes that sample, encode the changed native sample. Geometry-changing
augmentation cannot silently make location/GSD/time context valid; the project
method must update the context or disallow that augmentation. CutMix, for example,
may have no single meaningful geographic center.

## Retain per-sample inverse state

A recipe can retain an intermediate mean without passing it to the model:

```yaml
encoding:
  image:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref reflectance, dims: [band, time, y, x]}
  center:
    call: !ref image.mean
  centered:
    call: torch.sub
    kwargs: {input: !ref image, other: !ref center}
inputs:
  image: !ref centered
# After a model predicts a single output band on the input grid:
decoding:
  restored:
    call: torch.add
    kwargs: {input: !ref prediction, other: !ref center}
  raster:
    call: geosave.model.processing.to_raster
    kwargs:
      value: !ref restored
      reference: !ref optical
      dims: [band, y, x]
      name: prediction
```

This fragment assumes no further scaling of the model output. If input/output
normalization includes scaling, declare its inverse before restoring the center.
Decoding receives the values retained for this chip, even if other chips were
encoded in between. `to_raster` rejects an output whose spatial shape differs
from the reference grid; it does not guess a model's stride or valid interior.
It preserves grid placement, not input units, class names, packing, or nodata.
Declare output metadata explicitly. For a lower-resolution output, supply a
reference with its actual output grid.

For segmentation, merge logits before selecting classes:

```yaml
postprocessing:
  classes:
    call: !ref prediction.argmax
    kwargs: {dim: band}
```

```python
# Add one full logit array per chip, using the chip's global index.
chips.add(index, logits)
merged = chips.merge()
classes = spec.postprocess(merged[0])["classes"]
```

A model returning several outputs can bind them through ordinary mapping keys in
its decoding recipe. The caller decides which decoded values are merged and can
supply a mapping of merged outputs to `postprocess`.

## Run the end-to-end check

[Download the tested recipe](model_spec.yaml). It encodes a two-band, two-date
sample, retains a per-sample center, and restores a single-band regression output.
The tests load this exact file.

[The executable workflow test](https://github.com/weedkat/geosave/blob/main/tests/model/spec/test_processing.py)
creates two dated packed rasters, writes COGs and a GeoParquet catalog, samples
them, chips them, trains a tiny temporal regressor through a generated Lightning
DataModule, reloads its checkpoint and recipe, restores predictions using each
chip's mean, merges them, and reopens the saved output catalog. It compares the
result with independent NumPy arithmetic and checks laziness before encoding.

```bash
uv run pytest tests/model/spec/test_processing.py
```

## Migrating v2 recipes

- Set `schema_version: 3`.
- Replace `context:` with a named call in `encoding:`. Bind its returned keys in
  `inputs`, alongside explicitly encoded image tensors.
- Declare augmented argument names in `pixels`.
- Replace Kornia `name/init_args` entries in `transforms` with ordered calls.
  Fixed normalization uses `geosave.model.processing.normalize` and an explicit axis.
- Put training augmentation `size` in the DataModule config. There is no `chips`
  field in ModelSpec. Sampling and `Chips` remain explicit geodata operations.
- Replace `InputTransform` calls with `spec.transform`, stating `batched=True`
  when the DataLoader has added a batch axis.

Existing generated workspaces are copied project code; regenerate or migrate
that code explicitly. Existing trained weights retain their original band order
and normalization statistics.

## Catalog materialization

The existing `ingest`/`prepare` convenience methods remain available during this
iteration. `prepare` now selects collection nodes from nested catalog trees and
requires an exact common grid unless the caller supplies an alignment target.
It persists raster roots consumed before encoding, not arbitrary encoded state.
Recipes producing non-raster intermediates should use the in-memory execution
path. Automatic processing-scope inference and general intermediate persistence
are not implemented.
