# Chips and prediction merging

Sampling is three steps, each its own piece:

```python
windows = tree.gs.sample("MS", size=2, keep="last")     # time: one lazy tree per window
chipper = gs.RasterChipper(windows, size=256, stride=224)         # space: chips over every raster
merger = gs.RasterMerger(chipper.layout, window="hann")              # back: predictions by chip index

for index in range(len(chipper)):
    sample = chipper[index].gs.to_tensor()
    # {"image": (time, band, y, x), "label": ..., "transform": (6,), "crs": "EPSG:...", "time": (time,)}
    merger.add(index, predict(sample))                      # (band, y, x)

classes = merger.merge(0).argmax("band")                    # raster 0, on its own grid
```

`gs.sample` works on a Dataset, a DataArray or a DataTree with a datetime
`time` dimension. A slot of `freq` holding several observations keeps the
`first` or `last`; a window holding an empty slot is left out.

`RasterChipper` takes any list of one-grid trees: windows, several scenes, or both.
`mode` decides how chips meet a raster's edge:

| `mode` | Starts on a 10 px axis, size 4, stride 4 | |
| --- | --- | --- |
| `"reflect"` (default), `"edge"`, `"wrap"`, `"constant"` | 0, 4, 8 | last chip reaches past the edge and is filled as `numpy.pad` fills the whole raster |
| `"shift"` | 0, 4, 6 | last chip flush with the edge; no invented pixels |
| `"drop"` | 0, 4 | incomplete chips left out |

`RasterMerger` takes `chipper.layout`, weights overlaps with a supported scipy
window, and discards padding. It returns a georeferenced DataArray, with NaN
where uncovered. Full output accumulators remain in memory.

## Chips of tensor batches

The raster pair serves preparation and a release's processor. Inside a
LightningModule step, whose batch is a scene larger than the model's input,
the tensor pair chips and merges the same way: same `ChipLayout`, same window.

```python
chipper = gs.TensorChipper(batch, pixels=["inputs/image"], size=224, stride=192)
merger = gs.TensorMerger(chipper.layout, window="hann")

for start in range(0, len(chipper), 8):
    indices = range(start, min(start + 8, len(chipper)))
    chips = []
    for index in indices:
        chips.append(chipper[index])        # batch structure, transform cut to the chip
    chips = default_collate(chips)
    for index, logits in zip(indices, model(**chips["inputs"])):
        merger.add(index, logits)            # (band, size, size)

logits = merger.merge()                      # (batch, band, y, x), NaN where uncovered
```

| | Input | Chip | Merged |
| --- | --- | --- | --- |
| `RasterChipper` / `RasterMerger` | one-grid DataTrees, lazy | lazy DataTree | DataArray per raster |
| `TensorChipper` / `TensorMerger` | collated batch with `transform` and `crs` | sample dict, same keys | tensor, batch layout |

::: geosave.geodata.sampling.chipper.RasterChipper

::: geosave.geodata.sampling.layout.ChipLayout

::: geosave.geodata.sampling.layout.ChipLocation

::: geosave.geodata.sampling.layout.lay_chip_offsets

::: geosave.geodata.sampling.merger.RasterMerger

::: geosave.geodata.sampling.merger.create_chip_weights

::: geosave.geodata.sampling.tensor.TensorChipper

::: geosave.geodata.sampling.tensor.TensorMerger

::: geosave.geodata.core.raster.utils.windows.window_times
