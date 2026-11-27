# Pipeline data flow

A pipeline is four stages built in Python. The predictor takes the model
itself, so a released model is loaded first and passed in.

```python
from geosave.ml import Pipeline, Preprocessor
from geosave.ml.pipeline import Sentinel2L2AIngestor
from geosave.ml.pipeline.preprocessing import Composite, MaskAndScale
from geosave.ml.pipeline.postprocessing import SegmentationPostprocessor
from geosave.ml.torch import ModelChain
from geosave.ml.torch.prediction import TorchPredictor

model = ModelChain.from_pretrained("org/landcover", revision="v1.0")
pipeline = Pipeline(
    ingestors=[Sentinel2L2AIngestor(name="image", bands=["B02", "B03", "B04", "B08"], filter="eo:cloud_cover <= 40")],
    preprocessor=Preprocessor([MaskAndScale(), Composite(freq="MS", method="median")], time_freq="MS"),
    predictor=TorchPredictor(model, chip_size=224, chip_overlap=0.5),
    postprocessor=SegmentationPostprocessor(window="hann"),
)
classes = pipeline(region).product                     # in memory
classes = pipeline(region, store="runs/kenya").product # staged in Zarr, returned lazy
```

```text
<xarray.Dataset>
Dimensions:         (time: 12, y: 2048, x: 2048)
Coordinates:
  * time            (time) datetime64[ns] 2024-01-03 ... 2024-12-02   # each window's start
    start_datetime  (time) datetime64[ns] ...                         # each window's span
    end_datetime    (time) datetime64[ns] ...
  * y, x
    spatial_ref     int32
Data variables:
    class_id        (time, y, x) int32 dask.array<...>                # nodata -1

runs/kenya/
├── sources.zarr    # every collection, one tree, written in one computation
└── product/        # the merger's staged windows: 0.zarr, 1.zarr, ...
```

```python
class Pipeline:
    def __call__(self, region, store=None):
        ingested = self.ingest(region, store)               # IngestOutput: sources tree, memory or sources.zarr reopened
        preprocessed = self.preprocessor(ingested)          # PreprocessOutput: lazy trees, one per temporal window
        predicted = self.predictor(preprocessed)            # PredictOutput: the chipper and each chip's output
        return self.postprocessor(predicted, staged)        # PostprocessOutput: the product; staged = store / "product"
```

Each stage hands the next one output dataclass, the same for every
implementation of that stage, with the fields the next stage reads and an open
`metadata` dict for values a custom stage adds:

```python
IngestOutput(sources: xr.DataTree)
PreprocessOutput(windows: tuple[xr.DataTree, ...])          # refuses an empty tuple
PredictOutput(chips: RasterChipper, outputs: Iterator[(ChipLocation, output)])
PostprocessOutput(product: xr.Dataset | GeoDataFrame)
```

A stage checks what it reads from its input and says what it expected, such as
`RasterMerger needs dense chip outputs shaped (band, 128, 128); got (9,)`.

| Stage | Identified by | Input | Output |
| --- | --- | --- | --- |
| `Ingestor` | source collection | `GeoRegion` | its subtree under its name, lazy |
| `Pipeline.ingest` | — | region, optional store | `IngestOutput`: one tree, each collection a child of the root; in memory, or staged as `<store>/sources.zarr` and reopened |
| `Preprocessor` | model input | `IngestOutput` | `PreprocessOutput`: one prepared tree per temporal window; `Step`s run in order, one per `DataTree.gs` method |
| `Predictor` | ML framework | `PreprocessOutput` | `PredictOutput`: the chipper and an iterator of `(ChipLocation, output)`; it batches the chips and runs the model, as MONAI's `sliding_window_inference` does before its merge; a chip's `source` is its window |
| `Postprocessor` | task | `PredictOutput`, optional store | `PostprocessOutput`: the product over every window: a raster Dataset along `time` for dense tasks, a GeoDataFrame for detections; a dense one builds `RasterMerger(predicted.chips)` |

Contracts:

- A prepared tree's raster node paths are the model's input names; context
  readers add the rest. `TorchPredictor` passes them to the model by name.
- A step is named after the `DataTree.gs` method it runs and keeps that method's
  arguments. Representation order is enforced by the data (`composite` refuses
  packed values and numeric reducers on class IDs); semantic order, such as
  masking clouds before compositing, is the list order.
- The pipeline is standalone. Releases hold only the model; the predictor
  takes the loaded model, such as `ModelChain.from_pretrained(repo)`.
- A chip's location is the context of its output: its `source` states the
  window's position, grid and timespan, and `top`/`left` place the chip on that
  grid, so a postprocessor maps outputs to pixels or to geometries without the
  windows.
- A chip's output is the model's own structure as NumPy: one array for dense
  tasks, or several (boxes, scores, labels) for detections.
  `Postprocessor[Output, Product]` states which it reads and what it makes.
- Each window's product is dated by its timespan: `time` is its start, and
  `start_datetime`/`end_datetime` state the span (the bucket bounds when
  `write_time_bounds` stated them, else the first and last acquisition). A
  pipeline without temporal windows returns one timeless product.
- Zarr staging is internal to one run, never a publication. The sources are
  staged by the pipeline; a dense postprocessor's `RasterMerger` accumulates each
  window's sums and weights in `<store>/product/<window>.zarr`. Publishing products (COGs,
  STAC Items, catalogs) happens after the pipeline, in the flow.

Open: the publisher (where products go and how runs over several regions name
them), detection postprocessing and `TorchPredictor` splitting non-tensor
outputs, a cloud-masking step that reads one node to mask another, a
scikit-learn predictor, model outputs at another resolution than the input,
pipeline input from files as well as regions, resuming a run from its
store, and loading a released model from a pipeline YAML file (jsonargparse
cannot select the Hub mixin's `from_pretrained`, whose return type is generic).
