# Understand imagery and label values

Imagery stores measurements; labels store categories. Read their metadata so
you can calculate with reflectance without changing class codes.

**You need:** the base package, [training-image.tif](../assets/training-image.tif)
and [training-labels.tif](../assets/training-labels.tif).
[Data source](../assets/README.md#training-pair).

## Decode imagery to reflectance

```python
import geosave as gs

image = gs.open_raster("training-image.tif")
image["red"].gs.packing.scale_factor
```

```text
0.0001
```

The file stores integer measurements. Its scale says how to interpret them.
With scale `0.0001` and offset `0`, a stored value of `1000` represents
reflectance `0.1`. Calculations should use the measurement's meaning rather
than assuming the stored integer is already in physical units.

```python
reflectance = image.gs.mask_and_scale()
str(image["red"].dtype), str(reflectance["red"].dtype)
```

```text
('uint16', 'float64')
```

Decoding applies the declared scale and offset and turns missing pixels into
NaN. The source stays unchanged, and decoding can remain lazy. Spectral
indices such as NDVI already decode the bands they use.

The float dtype can represent both fractional reflectance and NaN. The file's
original uint16 storage information remains available for saving; decoding
does not rewrite the source TIFF.

## Keep labels as categories

```python
labels = gs.open_raster("training-labels.tif")
labels["label"].gs.flags.classes
```

```text
{0: 'water', 1: 'trees', 2: 'grass', 3: 'flooded_vegetation', 4: 'crops', 5: 'scrub', 6: 'built_area', 7: 'bare_ground', 8: 'snow_ice', 9: 'cloud'}
```

```python
labels["label"].gs.nodata
```

```text
255
```

Here, `0` is water and `255` is unannotated. Keep these integer codes separate
from reflectance. The legend also supplies the colors used when
[plotting the labels](training-chips.md#2-open-the-labels).

For training, the class legend defines what the model should predict. The
unannotated value is excluded from the loss; it is not an extra land-cover
class. Keep that choice consistent between the Dataset, loss and evaluation.

For your data, confirm the provider's units, scaling, class meanings and nodata
before preparing model inputs. See [metadata concepts](../concepts/metadata.md)
for preservation rules and the [metadata reference](../reference/attrs.md) for
writing declarations.
