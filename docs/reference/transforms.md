# Raster and vector transforms

These functions operate on native xarray and GeoPandas objects. Raster accessor
methods call the same operations. See [Raster grids](../concepts/raster-grids.md)
and [Metadata and pixel values](../concepts/metadata.md) for the behavior behind them.

## Grids

::: geosave.geodata.transform.warp
    options:
      members:
        - reproject
        - align
        - common_grid
        - Resampling
        - Target
        - Extent
        - Native

## Crop and pad

::: geosave.geodata.transform.crop
    options:
      members:
        - crop
        - window
        - Region
        - PaddingMode

## Combine rasters

::: geosave.geodata.transform.combine
    options:
      members:
        - merge_bands
        - concat_time
        - mosaic
        - Join
        - Compat
        - MosaicMethod

## Time

::: geosave.geodata.transform.temporal
    options:
      members:
        - resample
        - reduce
        - interpolate
        - Interpolation
        - Gap
        - Reducer

## Nodata and scaling

::: geosave.geodata.transform.nodata
    options:
      members:
        - mask
        - to_nan
        - unpack

## Rasterize and vectorize

::: geosave.geodata.transform.vector
    options:
      members:
        - rasterize
        - vectorize

## Class colors

::: geosave.geodata.transform.color
    options:
      members:
        - colorize
