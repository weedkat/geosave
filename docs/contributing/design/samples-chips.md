# Samples and chips

Updated against library source on 2026-11-15. Spatial layout, lazy cutting,
temporal windows and dense merging are implemented. Sample catalogs and store
orchestration below remain proposals. See [the sampling reference](../../reference/chips.md)
for callable APIs and [publish and load](publish-load.md) for catalog boundaries.

## Reading by chip GeoBox

A sample is a chip GeoBox, the time steps it covers and the sources read
onto it. Both ways of reading take the same grid. A historical prototype
reported identical pixels in a smoke test (4 time steps × 2 bands of 2048² COGs plus a 30 m
label on a shifted grid, local disk):

```python
# A. a prepared raster, already on one grid: crop
chip = scene.gs.crop_to_grid(chip_grid)                                    # 114 ms computed

# B. a catalog, sources on their own grids: ODC puts each on the chip grid
image = stac.load_dataset(s2_items, geobox=chip_grid, groupby="id", chunks=None)
label = stac.load_dataset(label_items, geobox=chip_grid, resampling="nearest", chunks=None)
                                                                           # 104 ms for both
```

| Use | When |
| --- | --- |
| A | image and label were ingested onto one grid (Dynamic World: label grid = image grid) |
| B | several sensors, time series, labels on another grid (S1 + S2 + labels) |

Neither needs a tree, a virtual cube, or a dask graph over the whole
catalog. Reprojection, resampling and mosaicking stay ODC's.

## Layout, cutting and dense merging

```python title="Adaptation example — use prepared imagery and labels on one grid"
tree = xr.DataTree.from_dict({"image": scene, "label": labels})
windows = tree.gs.sample("MS", size=2, keep="last")
chipper = gs.RasterChipper(windows, size=256, stride=224)
merger = gs.RasterMerger(chipper.layout, window="hann")
for index in range(len(chipper)):
    sample = chipper[index].gs.to_tensor()
    merger.add(index, predict(sample))  # NumPy (band, 256, 256)
merged = merger.merge(0)                # DataArray on raster 0's grid
```

`ChipLayout` owns input GeoBoxes, chip offsets, stride and edge placement.
`RasterChipper` holds one-grid DataTrees and cuts them lazily. `RasterMerger` owns
accumulators and shares the layout. The integer chip index preserves source
identity even when two rasters have the same grid; its location is available
through `chipper.layout.locate(index)`.

Temporal `.gs.sample` chooses one observation per calendar slot and returns
windows of slots. Empty slots exclude a window; `keep` is `first` or `last`.
Random training crops can use a directly sliced GeoBox and `crop_to_grid`.

Dense merging preserves the full source grid, removes padding, floors taper
weights at covered borders, and uses NaN for uncovered pixels. Predictions may
arrive in any order. The previous Tiler-backed class and GeoBox-only merger
proposal are superseded by this shared-layout implementation.

Open: accumulators occupy full arrays in memory. Larger outputs need streaming
writes or a merger per output region. Vector reconstruction, nearest-observation
slot selection and point-centred chip placement remain future work.

## Outputs per task

| Task | Model output per chip | Georeferenced result | Reconstruction |
| --- | --- | --- | --- |
| segmentation | logits `(class, y, x)` | uint8 DataArray with `CFFlags` from the classes | merge logits, then argmax |
| pixel regression | `(1, y, x)` | float DataArray with units, NaN nodata | merge values |
| object detection | boxes `(n, 4)` in chip pixels, scores, classes | GeoDataFrame in the chip CRS | `chip_grid.transform * (column, row)` per corner; concatenate; `torchvision.ops.batched_nms` over the overlaps |
| instance segmentation | masks | GeoDataFrame | `gs.vectorize` per chip, then dissolve across overlaps |
| chip classification | `(class,)` | GeoDataFrame, one row per chip footprint | none |

In a release, a registered postprocessor performs the decoding and
the `dense` predictor the dense reconstruction; see [release and inference](release-inference.md).
Dense results can be converted to dated Datasets and published through `Dataset.gs.to_catalog`;
vector results write through `.gs.to_geoparquet`.

Detection targets in training run the other way: clip the labels
GeoDataFrame to the chip extent, apply `~chip_grid.transform`, and collate
with a list collate function since box counts differ. `rows.gs.rasterize`
(currently a stub) covers burning labels onto a chip GeoBox.

## The Sample

Tree tensor conversion supplies named pixel tensors and, when requested,
grid and context entries. Each method owns its sample type; integration is
still being migrated between flat and nested samples in templates. Projects
retain chip indices for merging. For example:

```python
class Sample(TypedDict):          # declared in each method's data.py
    image: Tensor                 # (time, band, y, x) or (band, y, x)
    target: Tensor                # task-specific
    transform: Tensor             # float64 (6,): the chip GeoBox affine
    crs: Tensor                   # int64 EPSG code
    centroid: Tensor              # optional requested context: latitude, longitude
    year: Tensor                  # optional requested context: one per frame
    day_of_year: Tensor           # optional requested context: one per frame
```

```python
# anywhere after collation
grid = GeoBox(batch["image"].shape[-2:], Affine(*batch["transform"][i].tolist()), int(batch["crs"][i]))
```

| Fact (checked) | Consequence |
| --- | --- |
| `default_collate` stacks a float64 `(6,)` array to a float64 `(batch, 6)` tensor; the GeoBox rebuilt from the batch equals the original | no custom collate for dense tasks |
| float32 resolves 0.5 m at UTM northings | `transform` is never cast with the pixels |
| tensor grid conversion stores an int64 EPSG code | no WKT strings in collated samples; a CRS without an EPSG code is rejected |
| a dataclass does not collate | the Sample is a TypedDict, a plain dict at runtime |

Rules:

- Only what varies per sample travels with it. Band names, dtype, nodata,
  scale, classes are constant and live in the Collection next to the store.
- Geometric augmentation (flip, rotate) makes `transform` wrong. It runs in
  training only, on the GPU in the DataModule, where no georeferenced output
  is written. Validation, test and predict batches keep their grid.
- `TENSOR_CONTEXT` readers derive requested entries such as `centroid`,
  `year`, `day_of_year`, `week` and `hour` from the actual raster at tensor
  conversion. `DensePredictor` reads them per chip. TensorChipper adjusts
  the chip transform but retains existing context; deciding when to
  recompute location after augmentation remains open.

## Sample rows

Proposed, not implemented by catalog sampling: a sample is also an Item: the chip GeoBox as `proj:*`, the time window as
`start_datetime`/`end_datetime`, the source Items as properties. The
sample catalog is therefore an ordinary Item table (C1), written and read
by the same functions, and drawn by `rows.gs.explore()`.

```text
id            geometry   proj:code   proj:shape  proj:transform  start/end_datetime  split  s2            labels   valid_fraction
33TWL_12_07   POLYGON    EPSG:32633  [256, 256]  [10, 0, ...]    2025-06 .. 2025-09  train  [id, id, ""]  [id]     0.97
```

| Step | Mechanism | Owner |
| --- | --- | --- |
| lay chips over a region | `ChipLayout` → one row per chip (`projection.build_fields`) | library (W6) |
| attach source Items | `gpd.sjoin(chip_rows, item_rows)` | pandas, in the project or flow |
| time steps | group the joined rows by `pd.Grouper(key="datetime", freq=...)`; one list entry per step, `""` where a step has no Item | pandas; a small library function if two projects repeat it |
| choose what is worth training on | columns filled while preparing (`valid_fraction`, class counts), filtered with pandas; `WeightedRandomSampler` for balance | project |
| split | a `split` column from a spatial join against split polygons or MGRS squares, never a random row split of overlapping chips | project |
| row → GeoBox | `projection.read_geobox` (reads `proj:*` from an asset today; needs the same for a row) | library |

Open: how a time step is filled when several Items fall in it (first,
least cloudy, a composite). Raster `.gs.sample` supports first/last selection
and `.gs.composite` provides explicit reduction; catalog sample planning
still needs to record the chosen source Item ids. A named rule (a callable over the step's rows) is the
candidate if the release must state it for inference.

## Training data flow

```text
source catalog ─▶ sample rows ─▶ read each sample (A or B) ─▶ LitData store ─▶ StreamingDataset ─▶ DataModule ─▶ Module
   search,          plan             Sample dict                 + sample catalog     project Dataset     augment,
   ingest           (pandas)         per row                     + Collection         adds nothing        normalize (GPU)
└──── geosave_flows / project scripts, paid once ────────────────────────────┘ └───────── project, per epoch ─────────┘
```

| Stage | Reads | Writes | Owner |
| --- | --- | --- | --- |
| download / ingest | provider STAC | COGs + source catalog (publish-load) | flow task per region |
| plan | source catalog | sample rows | project function over pandas |
| prepare | sample rows, source catalog by path | LitData records (the Sample dict as numpy), sample catalog with the measured columns, Collection | flow; `litdata.optimize` over row positions |
| load | store | batches | `litdata.StreamingDataset` + `StreamingDataLoader` |
| augment, normalize | batch | batch | DataModule `on_after_batch_transfer` |

Findings kept from the prototype: workers open catalogs by path, never
receive them pickled; nothing runs at import in modules LitData or the
DataLoader re-imports; one stac-geoparquet file per collection.

Small projects skip the store: a torch Dataset over sample rows reading
with A or B is the same Sample, slower per epoch. The store is an
optimization, not a second contract.

Open: whether `litdata.optimize` keeps row order (the sample catalog must
line up with the store, or carry the record index). LitData is a base
dependency, but no library code writes or reads a store yet.
