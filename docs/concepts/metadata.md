# Metadata rules

Read a validated metadata value from an ordinary xarray band:

```python
import numpy as np
import xarray as xr

import geosave_engine as gs
from geosave_engine.geodata import attrs

band = xr.DataArray(np.array([[0, 1], [2, 255]], dtype="uint8"), dims=("y", "x"))
band = band.gs.write_nodata(255).gs.write_legend(
    {0: "background", 1: "forest", 2: "water"}
)
attrs.parse(band, attrs.Legend)
```

Output:

```text
Legend(flag_values=[0, 1, 2], flag_masks=None, flag_meanings='background forest water', color_map=None)
```

The attrs remain a plain xarray mapping. A group reads a few related keys as
one validated value; GeoSave keeps no second metadata store. Parsing an absent
group returns `None`. Parsing invalid metadata raises instead of guessing.
Use the `.gs` properties and `write_*` methods for everyday work, as shown in
[describe raster values](../basics/attrs.md).

## What the groups mean

| Group | Keys | Meaning |
| --- | --- | --- |
| `Nodata` | `_FillValue`; also reads `nodata` | Stored code for absent pixels. |
| `Packing` | `scale_factor`, `add_offset` | Conversion from stored to physical values. |
| `Legend` | `flag_values`, `flag_masks`, `flag_meanings`, `color_map` | Class codes or bit flags and their labels. |
| `Spectral` | `common_name`, `center_wavelength`, `full_width_half_max` | Band identity and wavelength in micrometres. |
| `GDALBand` | `variable_name`, `colorinterp` | Band naming and colour interpretation in GDAL formats. |
| `TimeSpec` | `time_freq`, `time_closed`, `time_label`, `time_origin`, `time_offset` | How the time coordinate describes periods. |

Band groups live on variables; `TimeSpec` lives on the `time` coordinate.
Other keys, such as `units`, `long_name`, or dataset `title`, stay ordinary attrs.

Writing a group removes all its previous keys before stating the new value.
This avoids stale aliases, and it validates values against the band's dtype
where they name pixel values. Writes copy metadata while sharing pixels;
they do not compute Dask arrays or recode classes.

## Join data with consistent meaning

```python
joined = xr.concat([band, band], dim="time", combine_attrs=attrs.combine)
joined
```

Output:

```text
<xarray.DataArray (time: 2, y: 2, x: 2)> Size: 8B
array([[[  0,   1],
        [  2, 255]],

       [[  0,   1],
        [  2, 255]]], dtype=uint8)
Dimensions without coordinates: time, y, x
Attributes:
    flag_values:    [0, 1, 2]
    flag_meanings:  background forest water
    _FillValue:     255
```

`attrs.combine` is an explicit xarray callback; ordinary xarray operations do
not automatically apply it. It keeps a group or plain key only when all inputs
state it alike.

When every input states nodata, packing, or a legend but the values disagree,
the join raises `ValueError`. Conflicting `units`, `standard_name`, and
`cell_methods` also raise when every input states the key. Make the pixel
meaning consistent before joining; changing an attr alone may be wrong if
the values actually use different scales or class codes.

If an input omits a group or key, the combined result drops it and warns.
Other differing attrs, including descriptions and spectral metadata, drop
with `DroppedAttrsWarning`. A warning therefore calls for review: it may be
an expected loss of per-file metadata, or a missing description needed later.

## Decode and encode are different stages

Raw stored values carry nodata and packing in `attrs`. CF file readers can move
these instructions to xarray `encoding` when opened with `mask_and_scale=True`.
The `.gs.mask_and_scale()` accessor removes the attrs but currently does not
retain that storage encoding. Do not assume its result will write back using
the original integer packing.

Keep the distinction between stored and physical values when editing metadata
or writing decoded data. Choose an output dtype, nodata value, and encoding
that describe the values you intend to save; verify that choice with a round
trip. Replacing NaNs or changing dtype is a pixel operation, not just an attr
edit.

GeoTIFF translates these groups into native band properties and tags. STAC
Items describe them through Raster, EO, and Classification metadata. A STAC
load restores the selected metadata that describes the loaded bands;
xarray attrs are not a lossless copy of the full Items or Collection.

Zarr records variable order in a storage attr and restores it on opening.
Metadata support varies by format, so verify a round trip for the properties
your workflow needs. See the [attrs reference](../reference/attrs.md) and
[read and write guide](../basics/io.md).
