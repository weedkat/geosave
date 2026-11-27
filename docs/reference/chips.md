# Chips and prediction merging

For training, `gs.random_crop` draws one window uniformly from all pixel
origins where the requested size fits. An integer seed repeats the draw on
the same grid. Omit it to draw a fresh window on each visit. Read a
segmentation target on the cropped image's grid:

```python
chip = gs.random_crop(image, size=128, seed=42)
target = label.gs.crop_to_grid(chip.gs.geobox)
```

## Sample time and space

```python
windows = gs.iter_samples(scene, "MS", size=2)
location, chip = next(iter(gs.RasterChipper(windows, size=256)))
```

Both APIs accept one DataArray, Dataset or DataTree, or an iterable of
them. Lists, tuples and one-pass generators all work. Passing `scene` gives
the same result as passing `[scene]`. A stream can mix raster types. Each
window and chip retains the type of its source.

`gs.iter_samples` selects windows within each raster's datetime `time` dimension.
Each time bucket keeps the first or last observation. A window containing
an empty bucket is skipped, or refused with `missing="raise"`. To blend a bucket's
observations or fill empty buckets, composite first with
`gs.composite(freq, method, interpolation=...)`.

In a DataTree, nodes observed on their own dates meet in the same buckets. Each
keeps its own observations, and static nodes accompany each window:

```text
s2   06-03, 06-28, 07-23        iter_samples(tree, "MS", size=2)    s2   06-28, 07-23
s1   06-10, 07-04, 07-16        ─────────────────────────────────▶  s1   06-10, 07-16
dem  static                                                          dem  static
```

`gs.RasterChipper` uses each input's own grid. A DataTree's raster leaves must
share one grid. The returned `ChipLocation` contains the input's position in
the stream and the chip's top and left pixel offsets. When the inputs are
temporal windows, that position is the window index.

Dask-backed pixels stay lazy during iteration. Calling `chip.gs.to_tensor()`
reads the source chunks needed by the chip, including any pixels used for padding.
For files, `chunks={}` selects Dask with the stored chunk sizes. A file opened
with `chunks=None` can read pixels while applying constant padding.

For model input, `chips.iter_batches(16)` yields `ChipBatch`es of `locations`
and loaded `chips`. It reads the raster one block of Dask chunks at a time and
cuts every chip that block completes, so each stored chunk is read once, or
twice when a chip straddles its lower or right edge. It holds one block in
memory; a raster without Dask chunks is read whole. `batch.to_tensor()` stacks
the chips' pixels as each `chip.gs.to_tensor()` returns them, with a batch
axis; `batch.to_tensor(context=...)` also stacks their context.

```python
for batch in chips.iter_batches(16):
    pixels, context = batch.to_tensor(context={"centroid": "centroid"})
```

An integer `overlap` counts shared pixels per axis. A float in `[0, 1)` gives
a fraction of the chip size, rounded down to whole pixels. For a 256 px chip,
`overlap=64` and `overlap=0.25` both place chip starts 192 px apart.
`constant_value` supplies the fill for constant padding.

| `mode` | Starts on a 10 px axis, size 4, overlap 0 | Edge behavior |
| --- | --- | --- |
| `"reflect"` (default), `"edge"`, `"wrap"`, `"constant"` | 0, 4, 8 | last chip reaches past the edge and is filled as `numpy.pad` fills the whole raster |
| `"shift"` | 0, 4, 6 | last chip sits flush with the edge |
| `"drop"` | 0, 4 | skips incomplete chips |

## Merge dense predictions

Build the merger from the chipper, as Tiler builds `Merger(tiler)`, and merge
every raster the chipper cut at once.

```python
chips = gs.RasterChipper(scenes, size=256, overlap=32)
merger = gs.RasterMerger(chips, window="hann")
for batch in chips.iter_batches(16):
    pixels, context = batch.to_tensor(context={"centroid": "centroid"})
    logits = model(image=pixels, **context).numpy()  # (chips, band, size, size)
    for location, chip_logits in zip(batch.locations, logits, strict=True):
        merger.add(location, chip_logits)

scores = merger.merge()  # tuple: one DataArray (band, y, x) per scene, in input order
classes = scores[0].argmax("band")
```

The DataLoader batch is the number of rasters; the model's batch is the
number of chips, and one chip batch can span rasters. A `RasterMerger`
averages dense chip predictions, such as class scores or pixelwise
regression. It takes the chipper's `sources`, each raster's position, `geobox`,
`timespan` and `extent`, and its chip size, so it knows every raster before
the first chip arrives. Each chip's `ChipLocation` names its source and its
top-left pixel: chips of any raster arrive in any order, and a location copied
through a DataLoader worker still names the same raster. A chip another
chipper cut, or an output not shaped `(band, size, size)`, is refused.
Overlaps are weighted by the chosen window and padding past the raster is
dropped. `merge()` returns one float32 DataArray per raster, on its grid with
its extent, NaN where no chip reached; a raster too small for one dropped chip
comes back all NaN in its place.

In memory, each raster's float32 sums and weights occupy `4 × (C + 1) × H × W`
bytes for `C` bands, `H` rows and `W` columns, and `merge()` divides into the
sums in place. For rasters larger than memory, give the merger a folder: raster
`i` accumulates in `<store>/<i>.zarr`, each chip adds into the chunks it
covers, and `merge()` returns DataArrays lazy over Dask. A Sentinel-2 tile
(10,980², nine classes, 3,249 chips of 256 px) merges in 0.6 GB of memory,
about 27 ms per chip, with 4.6 GB on disk.

```python
merger = gs.RasterMerger(chips, window="hann", store="runs/kenya/merged")
```

## Place a prediction on its grid

```python
from odc.geo.xr import wrap_xr

classes = wrap_xr(prediction.numpy(), image.gs.geobox).rename("class_id")
```

odc-geo's `wrap_xr` places a `(y, x)` array on a grid: the input raster's
`image.gs.geobox`, or a chip's `location.source.geobox`. For a channel-first
`(band, y, x)` array, name the axes:
`wrap_xr(scores, geobox, axis=1, dims=("band", *geobox.dimensions))`.
`RasterMerger.merge` already returns its rasters on their grids.

## Batch whole scenes in a DataLoader

```python
loader = DataLoader(scenes, batch_size=4, collate_fn=gs.collate)
images, labels = next(iter(loader))  # lists of four xarray objects
```

`gs.collate` batches samples as PyTorch's `default_collate` does, stacking
tensors, arrays and numbers, but keeps each DataArray, Dataset or DataTree as a
list, since scenes keep their own grids and sizes.

::: geosave.geodata.sampling.collate.collate

::: geosave.geodata.sampling.temporal.iter_samples

::: geosave.geodata.sampling.chipper.RasterChipper

::: geosave.geodata.sampling.chipper.ChipBatch

::: geosave.geodata.sampling.crop.random_crop

::: geosave.geodata.sampling.layout.ChipSource

::: geosave.geodata.sampling.layout.ChipLocation

::: geosave.geodata.sampling.layout.lay_chip_offsets

::: geosave.geodata.sampling.merger.RasterMerger


::: geosave.geodata.sampling.merger.create_chip_weights
