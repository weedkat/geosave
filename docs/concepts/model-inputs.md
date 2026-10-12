# Model input contracts

Read [model inputs](../guides/model-spec.md) first for a complete small example.
The spec separates the raster you accept, the processing you run on it, and
the arguments you give the model.

| Declaration | Consumer |
| --- | --- |
| `rasters` | `preprocess` selects and validates supplied Datasets; `load_rasters` acquires them first |
| `preprocessing` | `preprocess` runs the named calls in declaration order |
| `inputs` | `model_inputs` binds prepared rasters to model argument names |
| `context` | `model_context` encodes sample metadata; `model_inputs` adds those values |
| `frames`, `chips` | Explicit frame/chip cuts; training DataModules also read chip dimensions |
| `transforms` | Training DataModules apply configured Kornia transforms to pixel tensors |

The spec's processing methods do not run the model. `model_inputs` does not
convert raster pixels to tensors or apply `transforms`.

## References and calls

```yaml
preprocessing:
  reflectance:
    call: !ref optical.gs.mask_and_scale
  image:
    call: !ref reflectance.rename_vars
    kwargs:
      name_dict: {B04: red, B08: nir}
inputs:
  image: !ref image
```

These sections assume `rasters.optical.variables` declares `[B04, B08]`.
`!ref` looks up a supplied root and follows attributes or string keys. It does
not execute methods. The `call` field invokes the resolved callable with the
declared keyword arguments. It can name a bound method, as here, or an
importable module-level function such as `your_package.features.prepare`.

References can use `scene["view:sun_azimuth"]` for keys that are not Python
attribute names. In a YAML flow mapping, quote the path:
`{azimuth: !ref 'scene["view:sun_azimuth"]'}`. They are paths, not expressions:
function calls, numeric indexing, and arbitrary Python evaluation are not part
of reference syntax.

Preprocessing runs in declaration order and may read earlier results. Its
external roots must be declared rasters. Duplicate YAML keys and unknown spec
fields are rejected. Loading a spec runs no declared calls; invoking
preprocessing executes those Python functions.

## Require the right bands and metadata

```yaml
rasters:
  optical:
    variables: [B04, B08]
    coordinates: [x, y, time]
    dims: [time, y, x]
    dtypes: [uint16]
    require_crs: true
    resolution: 10
    attrs:
      data_vars:
        "*":
          required: [scale_factor]
          equals: {scale_factor: 0.0001}
```

`variables` selects named data variables in the given order. For a source
whose names have no band meaning, `channels: 2` selects its first two data
variables instead. Exactly one of `variables` and `channels` is required.

`coordinates` requires coordinate arrays; it does not constrain their values.
`dims` requires the exact dimension order for every selected variable. `dtypes`
checks stored dtype, and `resolution` checks absolute pixel size in CRS units.
These checks validate the raster as supplied, before preprocessing, and do not
resample, cast, or read pixels to repair it.

Attribute requirements can apply to `root`, `data_vars`, and `coords`.
`required` demands a non-null key, `equals` checks a value, and `one_of` permits
one of a list of values. `"*"` applies to all selected variables or all
coordinates in its scope. Comparisons decode serialized attribute text, so a
GeoTIFF tag containing `"0"` can satisfy an expected numeric `0`.

## Acquire declared rasters from STAC

```yaml
rasters:
  optical:
    variables: [B04, B08]
    stac:
      collection: sentinel-2-l2a
      endpoints:
        - https://planetarycomputer.microsoft.com/api/stac/v1/
      query:
        datetime: ["2025-01-01", "2025-01-31"]
        max_items: 8
      load:
        bands: [B04, B08]
        chunks: {x: 512, y: 512}
```

Call `spec.load_rasters(anchor)` with a `GeoAnchor` to acquire and validate the
rasters on its grid. Every declared raster needs a `stac` block for this method;
the ordinary `preprocess` path can use local rasters without recipes.

Named `load.bands` must match `variables` in order. Explicit query extent and
datetime take priority; the anchor supplies omitted selections. Endpoints are
tried in order when collection resolution fails. Once an endpoint supplies
the collection, a search or loading failure propagates. See [STAC](../guides/stac.md)
for native search and loading APIs.

## Normalize pixel tensors

Add these sections to a spec whose `inputs.image` reads two prepared bands:

```yaml
chips:
  size: 224
  overlap: 32
  mode: reflect
  window: hann
transforms:
  image:
    - name: Normalize
      init_args:
        mean: [0.15, 0.27]
        std: [0.19, 0.17]
```

`preprocessing` acts on native rasters, usually before storing prepared data.
`transforms` declares Kornia operations on pixel tensors. Training DataModules
apply them after training augmentation, and also in validation and test.
Per-band normalization applies separately to each temporal frame. Context
values pass through these pixel transforms unchanged.

For a direct model call, apply the corresponding tensor transform explicitly:

```python
from kornia.augmentation import Normalize

normalize = Normalize(mean=[0.15, 0.27], std=[0.19, 0.17])
image = normalize(image)
```

Here `image` has shape `(batch, band, y, x)`. A saved model wrapper does not
automatically apply the spec's transforms. Match the training preparation when
using a release for inference.

`frames` and `chips` declare window cuts; `prepare` does not apply them.
Use `spec.frames.cut(windows)` when frames are declared, then
`spec.chips.cut(windows)` for spatial chips. A merge taper such as `hann` needs
overlap; use at least two shared pixels for the tapers with zero edge weights.

## Add geospatial context

Prithvi TL and Clay use acquisition time and location alongside image pixels.
The context recipe reads the prepared raster's metadata.

```yaml
inputs:
  image: !ref image
context:
  call: geosave.model.stages.encoder.prithvi.model_context
  kwargs: {raster: !ref image}
```

| Encoder | Context returned for one sample |
| --- | --- |
| `PrithviTL` | `temporal_coords` shaped `(time, 2)`: year and zero-based day of year; `location_coords` shaped `(2,)`: latitude and longitude |
| `Clay` | `time` shaped `(2,)`: ISO week and hour; `latlon` shaped `(2,)`: latitude and longitude |

For Clay, use `geosave.model.stages.encoder.clay.model_context` and a raster
with one acquisition instant. The functions return float32 tensors and read
time labels and the grid centre, without loading pixels. Missing time or grid
metadata raises.

`spec.model_inputs(rasters)` resolves pixels and adds this context dictionary.
`spec.model_context(rasters)` can encode it separately, then
`spec.model_inputs(rasters, context=cached)` reuses it. Context may read only
roots already read by `inputs`, because `prepare` stores those roots. Context
keys cannot overwrite pixel input names.

Encode context while the sample still has raster coordinates. Convert the
pixel rasters to tensors afterwards; batching stacks the context tensors along
with the pixel tensors.
