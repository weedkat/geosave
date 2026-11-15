# API reference

Find a call by what you want to do, then open its signature for parameters,
defaults, return values and errors. GeoSave operates on native xarray and
GeoPandas objects; their own methods retain upstream behavior.

## Find an operation

| I want to… | Call and reference |
| --- | --- |
| Open raster or vector files | `gs.open_raster`, `gs.open_vector`. [Readers and format options](io.md) |
| Inspect a pixel grid | `raster.gs.geobox`. [Dataset accessor](dataset-accessor.md#geosave.geodata.core.raster.dataset.GeoDataset.geobox) |
| Crop by geometry or exact grid | `.gs.crop`, `.gs.crop_to_grid`. [Crop and grid operations](transforms.md#grid-masks-and-values) |
| Match another image's grid | `.gs.reproject(...)`. [Reprojection](dataset-accessor.md#geosave.geodata.core.raster.dataset.GeoDataset.reproject) |
| Decode nodata, scale and offset | `.gs.mask_and_scale()`. [Value metadata](../basics/attrs.md#decode-stored-measurements) |
| Calculate NDVI or another index | `gs.features.ndvi(...)`. [Features and masks](features.md) |
| Convert selected bands to arrays or tensors | `.gs.to_numpy`, `.gs.to_tensor`. [Dataset conversion](dataset-accessor.md#geosave.geodata.core.raster.dataset.GeoDataset.to_numpy) |
| Window time, cut chips or merge predictions | `.gs.sample`, `gs.RasterChipper`, `gs.RasterMerger`. [Chips](chips.md) |
| Reduce observations into calendar bins | `.gs.composite`. [Composites](rasters.md#composites-and-time-selection) |
| Save rasters and schedule writes | `io.write_cog`, `.gs.to_zarr`, `.gs.to_netcdf`. [Writer returns](io.md#saved-raster-files) |
| Search observations or load STAC Items | `gs.StacClient`, `stac.load_dataset`, `stac.group_bands_by_grid`. [STAC](stac.md) |
| Publish a dated Dataset | `Dataset.gs.to_catalog(...)`. [Catalog operations](catalog.md) |
| Build a model from stages | `geosave.ml.build_model`. [Chains and registry](model-chain.md) |
| Save and load model weights | `ChainModel`. [Releases](model-release.md) |
| Predict a raster with a released model | `Processor`, `build_processor`, `load_processor`, `ChipPredictor`. [Processors](model-release.md#processors) |

## Look up methods for your object

For `scene.gs.reproject(...)`, use the [Dataset accessor](dataset-accessor.md).
For a band, use the [DataArray accessor](dataarray-accessor.md); for a hierarchy,
use the [DataTree accessor](datatree-accessor.md). [GeoDataFrame methods](vectors.md)
are separate. Inherited GeoSave methods are included on each accessor page.

| Area | APIs |
| --- | --- |
| [Data](geodata.md) | Accessors, readers, writers, features, metadata and STAC. |
| [Models](model-chain.md) | Chains, [stages](model-stages.md) and native [releases](model-release.md). |
| [Training](ml.md) | Losses, metrics, transforms, callbacks and Lightning tools. |
| [CLI](cli.md) | Workspace commands and generated files. |
| [Optional flows](workflow.md) | Prefect helpers and current availability. |

## Read a signature

Parameters show types and defaults. Descriptions explain accepted values,
units, array axes, variable order, mutation and pixel execution. TypedDicts
list optional backend keywords; Pydantic fields are constructor keyword inputs.
Follow [format options](io.md) when a call forwards `**options`.

The [symbol inventory](public-api.md) also lists low-level module definitions,
configuration fields and protocols. Use it when you know a symbol's name;
start with the operations above for everyday calls.

For complete workflows, use [Find a task](../activities.md). For behavior shared
across methods, use [Concepts](../concepts/index.md).
