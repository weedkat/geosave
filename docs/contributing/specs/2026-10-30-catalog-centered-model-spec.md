# Catalog-centered model spec

## Intent

Each processing flow produces one catalog. A catalog keeps the metadata of
every raster it lists, combines with `GeoCatalog.concat`, and feeds `sample`
the same way for training and inference, so it is the unit processing is
organized around: every step takes a catalog and gives one.

```text
           ingest(anchor) ─┐
create_catalog(own files) ─┼─► raw catalog ─prepare─► prepared catalog ─sample─► GeoSamples ─(model)─► postprocess ─► prediction catalog
       open_vector(table) ─┘
```

Training is the prepared catalog plus label Items, carried through `prepare`
with `keep`. Inference uses the prepared catalog alone. A served model
(LitServe) receives a raw catalog or an anchor.

## Terms

| Term | Step | Takes -> gives |
| --- | --- | --- |
| ingest | `ModelSpec.ingest(anchor, path)` | anchor -> raw catalog of the declared rasters, loaded from their STAC recipes and saved as published |
| prepare | `ModelSpec.prepare(raw, path=None, *, target=None, keep=())` | raw catalog -> prepared catalog: every sample's declared rasters selected, validated and preprocessed, the rasters `inputs` reads saved as COGs |
| sample | `ModelSpec.sample(catalog, *, target=None)` | prepared catalog -> GeoSamples, framed and chipped as declared |
| postprocess | later spec | GeoSamples + model outputs -> prediction catalog |

`ingest` is the geospatial term for bringing data into a catalog; "acquire"
already names a satellite's acquisition, and "fetch" says nothing geospatial.
`prepare` without a path writes to a temporary folder, since prepared rasters
serve one model; a path keeps them. `prepare` does not frame: time bins select
among prepared rasters, including composites dated by their span, in `sample`.

`FramesSpec` and `ChipsSpec` keep their fields and validation; their `cut`
methods fold into `ModelSpec.sample`, which calls `GeoSamples.frame` and
`GeoSamples.chip` directly.

## Flow

The `geosave-flows ingest` flow writes the raw catalog as one
`catalog.parquet` instead of a sample directory, with minimal Collection
definitions named after the declared rasters.

## Not in scope

Postprocessing and the prediction catalog; Collection definitions derived from
loaded rasters; tensor transforms outside `geosave.ml`; output grids that differ
from chip grids.

## Breaking changes

`FramesSpec.cut` and `ChipsSpec.cut` are removed; the ingest flow and CLI return
a catalog path and drop `--format` / `--write-options`; `write_sample` is
removed.
