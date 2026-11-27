# Align labels with imagery

Put an existing label raster on the exact grid of an image before training.
Equal array shapes or matching CRSs alone do not establish alignment.

**You need:** the base package, a georeferenced image and a label TIFF.
Replace paths and band names below with your files. Label values must already
be integer class IDs, with their nodata value declared correctly.

```python title="Adaptation example — replace paths and band names"
import geosave as gs
from geosave.geodata import io

image = gs.open_raster("data/image.tif")
labels = gs.open_raster("data/labels.tif")
aligned = labels.gs.reproject(image.gs.geobox, resampling="nearest")
path = io.write_cog(aligned, "data/aligned-labels.tif")
print(path)
```

```text
data/aligned-labels.tif
```

The reference image decides the output CRS, pixel size, extent and alignment.
Nearest-neighbor resampling preserves class IDs. Bilinear or averaging
kernels can be suitable for continuous imagery, but cannot be used to blend
categorical labels. Reproject image and labels separately when they need
different kernels.

Alignment can stay lazy until writing. For model training, read
[chips](overlapping-chips.md) from aligned imagery and labels.

## Missing coverage and target class IDs

Reprojection does not create labels beyond source coverage. Choose a nodata
value that fits the label dtype, such as 255 for uint8, and declare it before
reprojection with `.gs.write_nodata(255)` if the source lacks the correct
metadata. This declares missingness; it does not replace existing pixel values.

Before training, map valid class IDs to `0 … num_classes - 1`, and configure
the loss to ignore the missing-label value. Keep labels separate from decoded
imagery so their class and missing-value metadata retain their meaning.

## Changing resolution or CRS

To choose a new analysis grid, construct a GeoBox with the required CRS,
resolution and extent, then pass that same GeoBox to both reprojections.
Passing only a CRS lets each source derive its own grid and does not ensure
that the resulting pixels align.

See [raster grids](../concepts/raster-grids.md) for grid definitions and
[the reprojection reference](../reference/dataset-accessor.md#geosave.geodata.core.raster.dataset.GeoDataset.reproject)
for accepted arguments.
