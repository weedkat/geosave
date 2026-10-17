# GeoSave

GeoSave makes geospatial data feel like normal Python. Open an image, work
with its bands, and save results that still know where they belong on Earth.
When you're ready, use that same data to build and train a model.

Its strength is keeping the meaning of your data attached: pixel locations,
band information, and missing values. You work with familiar Python objects,
and large images can stay lazy so you can process the part you need.

## A first look

For a georeferenced image with bands named `red` and `nir`:

```python
import geosave as gs

image = gs.open_raster("scene.tif")
ndvi = gs.features.ndvi(image, nir="nir", red="red").rename("ndvi")
ndvi.gs.to_cog("ndvi.tif")
```

This calculates a vegetation index and saves a new image on the same grid.
`image` is a normal xarray Dataset. GeoSave adds geospatial
operations through `.gs`, so your data stays usable with the wider Python
ecosystem.

## Start small

For a complete research learning path, follow
[Learn geospatial AI by example](examples.md): train on satellite image/label
pairs, swap model stages, improve the inputs, and put predictions back on a map.
The notebooks use ordinary PyTorch and Lightning.

1. [Install GeoSave](install.md).
2. [Try your first raster](first-raster.md): make a tiny image, calculate an
   index, and save it. No downloads or data files needed.
3. [Explore the basics](basics/index.md): images, polygons, missing values,
   and file operations.

## Find your task

| I want to… | Start here |
| --- | --- |
| Open and prepare my files | [Ingest files](guides/ingest-files.md) |
| Find satellite imagery | [Search and load imagery](guides/stac.md) |
| Learn geospatial model development through a complete project | [Notebook learning path](examples.md) |
| Turn imagery and labels into training chips | [Images to training chips](quickstart.md) |
| Build a model using pretrained weights | [Pretrained models](guides/pretrained.md) |
| Train a model on my data | [Training](guides/training.md) |
| Save and reuse a trained model | [Model releases](guides/release.md) |

See [all activities](activities.md) for more tasks. Read
[Advanced](concepts/index.md) when you need to understand the data flow,
grids, or training behavior. Use [Common questions](faq.md) and the
[API reference](reference/index.md) to look up details.

GeoSave is in Alpha; these docs follow the current implementation.
