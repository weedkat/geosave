# Describe raster values

Give a label raster its class names and a missing-data value:

```python
import numpy as np

import geosave as gs

anchor = gs.GeoAnchor.from_coordinates(
    latitude=45, longitude=12, shape=4, resolution=10, crs="EPSG:32633"
)
codes = np.array([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
    [1, 1, 2, 2],
    [1, 2, 2, 255],
], dtype="uint8")
labels = gs.raster({"label": (("y", "x"), codes)}, anchor.geobox, nodata=255)
labels = labels.gs.write_legend(
    {0: "background", 1: "forest", 2: "water"}, target="label"
)
labels["label"]
```

Output:

```text
<xarray.DataArray 'label' (y: 4, x: 4)> Size: 16B
array([[  0,   0,   1,   1],
       [  0,   1,   1,   2],
       [  1,   1,   2,   2],
       [  1,   2,   2, 255]], dtype=uint8)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
Attributes:
    _FillValue:     255
    flag_values:    [0, 1, 2]
    flag_meanings:  background forest water
```

The array stores class codes. The legend states what those codes mean, and
`_FillValue=255` marks an absent label. Background remains a valid class here.
Metadata lives in the xarray object's `attrs`; GeoSave validates the keys used
for geospatial operations and file formats.

## Read the meaning of a band

```python
labels["label"].gs.legend.classes
```

Output:

```text
{0: 'background', 1: 'forest', 2: 'water'}
```

```python
labels["label"].gs.nodata
```

Output:

```text
255
```

For a `Dataset`, `.gs.nodata` returns a mapping keyed by variable name because
different variables can use different fill values. `.gs.write_nodata(value,
target="label")` changes one variable; omit `target` to change all of them.

Writing nodata or a legend changes metadata, not pixel values. The fill value
and class codes must fit the stored dtype: `255` fits `uint8`, while `-1` does
not. Do not choose a valid class as nodata unless you intend to exclude it.

## Decode stored measurements

Some files store integers and describe the physical values with a scale and
offset. Decode before using those values in your own calculations:

```python
stored = gs.array(
    np.array([[0, 1000], [2000, 3000]], dtype="uint16"),
    anchor.geobox[:2, :2],
    dims=("y", "x"),
    nodata=0,
).gs.write_packing(scale_factor=0.0001)
reflectance = stored.gs.mask_and_scale()
reflectance.compute()
```

Output:

```text
<xarray.DataArray (y: 2, x: 2)> Size: 16B
array([[       nan, 0.09999999],
       [0.19999999, 0.29999998]], dtype=float32)
Coordinates:
  * y            (y) float64 16B 4.987e+06 4.987e+06
  * x            (x) float64 16B 2.635e+05 2.635e+05
    spatial_ref  int32 4B 32633
```

`mask_and_scale()` masks nodata first, then applies
`stored * scale_factor + add_offset`. The result uses floating-point values
and NaN for missing pixels. It preserves lazy arrays and leaves `stored`
unchanged. The result no longer carries nodata and packing attrs. This accessor
does not retain the original storage encoding; when saving decoded values,
choose the output dtype, nodata, and encoding explicitly.

Readers return stored values by default. Use `gs.open_raster(path,
mask_and_scale=True)` to decode while opening. Spectral indices already decode
the bands they select; cloud-mask functions expect decoded reflectance.

## Add descriptive metadata

```python
labels["label"].attrs["long_name"] = "Land cover classes"
labels["label"]
```

Output:

```text
<xarray.DataArray 'label' (y: 4, x: 4)> Size: 16B
array([[  0,   0,   1,   1],
       [  0,   1,   1,   2],
       [  1,   1,   2,   2],
       [  1,   2,   2, 255]], dtype=uint8)
Coordinates:
  * y            (y) float64 32B 4.987e+06 4.987e+06 4.987e+06 4.987e+06
  * x            (x) float64 32B 2.635e+05 2.635e+05 2.636e+05 2.636e+05
    spatial_ref  int32 4B 32633
Attributes:
    _FillValue:     255
    flag_values:    [0, 1, 2]
    flag_meanings:  background forest water
    long_name:      Land cover classes
```

Ordinary keys such as `long_name`, `units`, and dataset `title` remain ordinary
xarray attrs. Their preservation depends on the operation and output format;
adding attrs does not change values or coordinates.

Read [metadata rules](../concepts/metadata.md) for validated groups, joins, and
format boundaries. See the [attrs API](../reference/attrs.md) for direct parsing
and writing helpers.
