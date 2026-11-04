# Model input contracts

Read [model inputs](../guides/model-spec.md) first for a complete small example.
The spec separates the raster you accept, the processing you run on it, and
the arguments you give the model.

| Declaration | Consumer |
| --- | --- |
| `rasters` | `preprocess` selects and validates supplied Datasets |
| `preprocessing` | Ordered native processing of the supplied sample |
| `encoding` | Tensor conversion, metadata encoding, and retained inverse state |
| `inputs` | `model_inputs` binds only the declared arguments |
| `pixels` | Names the image arguments a training method may augment |
| `tensors` | `transform` checks sample shape and dtype, including shared symbolic axes |
| `transforms` | `transform` applies the same calls in training and inference |
| `decoding` | `decode` processes one prediction with that sample's state |
| `postprocessing` | `postprocess` processes merged outputs |

ModelSpec does not execute the model, choose observations, or lay out chips.
A stage runs only when invoked. See the [execution guide](../guides/model-spec.md)
for a complete v3 recipe and the catalog-to-training smoke test.

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

Call `spec.load_rasters(anchor, geobox=grid)` with a `GeoAnchor` to acquire and validate the
rasters on its grid. Every declared raster needs a `stac` block for this method;
the ordinary `preprocess` path can use local rasters without recipes.

Named `load.bands` must match `variables` in order. Explicit query extent and
datetime take priority; the anchor supplies omitted selections. Endpoints are
tried in order when collection resolution fails. Once an endpoint supplies
the collection, a search or loading failure propagates. See [STAC](../guides/stac.md)
for native search and loading APIs.

## Normalize pixel tensors

```yaml
transforms:
  image:
    call: geosave.model.processing.normalize
    kwargs:
      value: !ref image
      mean: [0.15, 0.27]
      std: [0.19, 0.17]
      dim: -3
```

Here image ends in `(band, y, x)`. The explicit negative axis works with or
without a leading batch or temporal axis. Generated DataModules run
`spec.transform(inputs, batched=True)` after training augmentation and in
validation/test. Direct inference calls the same method, using `batched=False`
for an unbatched sample. Nothing normalizes a model call automatically.

## Encode geospatial context

```yaml
encoding:
  image:
    call: geosave.model.processing.to_tensor
    kwargs: {raster: !ref reflectance, dims: [time, band, y, x]}
  context:
    call: geosave.model.stages.encoder.prithvi.model_context
    kwargs: {raster: !ref reflectance}
inputs:
  image: !ref image
  temporal_coords: !ref 'context["temporal_coords"]'
  location_coords: !ref 'context["location_coords"]'
pixels: [image]
```

Context can read any declared/prepared raster, including auxiliary metadata
that is not a pixel input. The same named-value mechanism encodes pixels and
context; model arguments are explicitly bound. Encode after changing the
sample's native crop or temporal selection. Never reuse cached context for a
different sample.

Prithvi returns temporal coordinates `(time, 2)` and location `(2,)`. Clay's
installed `model_context` returns `time` and `latlon`, both `(2,)`, and requires
one instant. These functions inspect coordinates, not pixels. Wavelengths and
GSD remain part of the encoder configuration and must match input semantics.

Geometry-changing training augmentation still requires a project-specific
context policy. The generated image augmenters do not update geospatial context
and assume trailing `(band, y, x)` image axes.
