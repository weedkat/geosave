# Describe raster values

Give a label raster its class names and a missing-data value:

```python
import numpy as np
import xarray as xr
from odc.geo.xr import xr_coords

import geosave as gs

from odc.geo.geobox import GeoBox

grid = GeoBox.from_bbox(
    (500000, 5000000, 500040, 5000040),
    crs='EPSG:32633', resolution=10,
)
codes = np.array([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
    [1, 1, 2, 2],
    [1, 2, 2, 255],
], dtype="uint8")
labels = xr.Dataset({"label": (("y", "x"), codes)}).assign_coords(xr_coords(grid)).gs.write_nodata(255)
labels = labels.gs.write_flags(
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
  * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
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
labels["label"].gs.flags.classes
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

| Operation | Input | Result |
| --- | --- | --- |
| `.gs.write_nodata(value)` | A fill value representable in each selected dtype | New object with nodata attrs; existing pixels keep their values. |
| `.gs.write_flags(classes)` | Integer codes mapped to names | New object with class metadata; existing pixels keep their values. |
| `.gs.mask(valid)` | Boolean mask, `True` where pixels are valid | New object with invalid pixels replaced by each variable's nodata. |
| `.gs.mask_and_scale()` | Stored values with optional nodata and packing | New object with missing values masked and packing decoded. |

The writers return a new xarray object; assign their result to keep the change.
Masking needs a declared nodata value or an explicit `fill=`. Attribute writers
do not find, replace, or validate every pixel in the raster.

## Decode stored measurements

Some files store integers and describe the physical values with a scale and
offset. Decode before using those values in your own calculations:

```python
stored = (
    xr.DataArray(np.array([[0, 1000], [2000, 3000]], dtype="uint16"), dims=("y", "x"))
    .assign_coords(xr_coords(grid[:2, :2]))
    .gs.write_nodata(0)
    .gs.write_packing(attrs.CFPacking(scale_factor=0.0001))
)
reflectance = stored.gs.mask_and_scale()
reflectance.compute()
```

Output:

```text
<xarray.DataArray (y: 2, x: 2)> Size: 32B
array([[nan, 0.1],
       [0.2, 0.3]])
Coordinates:
  * y            (y) float64 16B 5e+06 5e+06
  * x            (x) float64 16B 5e+05 5e+05
    spatial_ref  int32 4B 32633
```

`mask_and_scale()` masks nodata first, then applies
`stored * scale_factor + add_offset`. The result uses floating-point values
and NaN for missing pixels. It preserves lazy arrays and leaves `stored`
unchanged. The consumed nodata, scale and offset move from variable attrs to
variable encoding, alongside the original storage dtype. Writers use that
encoding to restore the stored representation. Supply explicit output encoding
to choose a different representation. No manual reverse conversion is needed.
Decoding expects native CF names and numeric values. Use `write_nodata` for a
manually constructed array: it delegates dtype conversion to rioxarray, so the
fill is stated at the array's precision. Direct attrs are not repaired during
decoding.

Readers return stored values by default. Use `gs.open_raster(path,
mask_and_scale=True)` to decode while opening. STAC loading retains DN by default; use
`load_dataset(..., mask_and_scale=True)` for immediate in-memory decoding, or
`.gs.mask_and_scale()` on a loaded Dataset or DataArray. Dataset decoding also
interprets CF coordinates and relationships. Spectral indices already decode
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
  * y            (y) float64 32B 5e+06 5e+06 5e+06 5e+06
  * x            (x) float64 32B 5e+05 5e+05 5e+05 5e+05
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

GeoSave's Dataset-to-DataArray and DataTree-to-Dataset conversions accept
xarray's `combine_attrs` policies. Their default, `"drop_conflicts"`, keeps
nonconflicting keys, including keys present in only one input, and automatically
warns about dropped keys. Use `combine_attrs="override"` to take the first
input's attrs, or `"identical"` to require agreement. Attr groups validate
declarations; they do not change how xarray combines keys.

For ordinary concatenation and merging, call `xr.concat` or `xr.merge` directly.
Their default is `"override"`; pass `combine_attrs=gs.attrs.combine_attrs` for
GeoSave's drop-conflicts policy and warnings. Native joins do not check that
stored pixels share packing, CRS, units or class meanings. Decode and align
inputs explicitly before combining them. GeoSave's band stacking still requires
compatible active CF packing.

Dataset-level identity and discovery fields use ACDD metadata, available as
`raster.gs.acdd` and written with `raster.gs.write_acdd(gs.attrs.ACDD(...))`.
Band-level nodata, packing, spectral facts, and legends stay on the variables.
When loading STAC, Collection metadata is retained as `StacCollection` on the
raster root, while each Asset's band metadata describes its variables.

Read [metadata rules](../concepts/metadata.md) for validated groups, joins, and
format boundaries. See the [attrs API](../reference/attrs.md) for direct parsing
and writing helpers.
