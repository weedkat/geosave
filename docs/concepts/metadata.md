# Metadata rules

Read a validated metadata value from an ordinary xarray band:

```python
import numpy as np
import xarray as xr

import geosave as gs
from geosave.geodata import attrs

band = xr.DataArray(np.array([[0, 1], [2, 255]], dtype="uint8"), dims=("y", "x"))
band = band.gs.write_nodata(255).gs.write_legend(
    {0: "background", 1: "forest", 2: "water"}
)
attrs.Legend.from_attrs(band.attrs)
```

Output:

```text
Legend(flag_values=[0, 1, 2], flag_masks=None, flag_meanings='background forest water', color_map=None)
```

The attrs remain a plain xarray mapping. A group reads a few related keys as
one validated value; GeoSave keeps no second metadata store. Reading an absent
group returns `None`. Reading invalid metadata raises instead of guessing.
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
| `ACDD` | `id`, `title`, `summary`, `keywords`, `institution`, `license` | Dataset identity and discovery metadata on root attrs. |

Band groups live on variables; `TimeSpec` lives on the `time` coordinate.
Other keys, such as `units` and `long_name`, stay ordinary attrs. ACDD keys
also stay in that mapping; the group reads their dataset-level meaning.

Write a group with `attrs.stamp_attrs(obj, value)` and remove one with
`attrs.clear_attrs(obj, Group)`. Change some fields of a stated group with
`Group.replace(obj.attrs, field=...)`, then stamp the result. Each returns a
shallow copy of the object, so the original keeps its attrs.

```python
band = attrs.stamp_attrs(band, attrs.Nodata(value=255))
band = attrs.clear_attrs(band, attrs.Nodata)
band = attrs.stamp_attrs(band, attrs.GDALBand.replace(band.attrs, colorinterp="red"))
```

Writing a group removes all its previous keys before stating the new value.
This avoids stale aliases, and it validates values against the band's dtype
where they name pixel values. Writes copy metadata while sharing pixels;
they do not compute Dask arrays or recode classes.

## Join data with consistent meaning

```python
joined = xr.concat([band, band], dim="time", combine_attrs=attrs.combine_attrs)
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

`attrs.combine_attrs` is an explicit xarray callback; ordinary xarray operations do
not automatically apply it. xarray calls it once per variable, coordinate and
root, with that place's attrs from every input. It keeps a group or plain key
only when all inputs state it alike. `attrs.intersect` does the same without
refusing or warning, as when `gs.to_array` stacks the bands of one raster.

`ACDD` joins each field separately, so a shared license can remain even when
dataset titles differ. The warning and conflict rules below still apply to
the other groups and plain keys.

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

STAC observation properties requested with `with_properties` retain ODC's
representation as variables, using original names such as `eo:cloud_cover`.
Use `.set_coords("eo:cloud_cover")` explicitly to carry a property as metadata
when writing a Scene. Recognized scalar coordinates are exported automatically
when that Scene becomes an Item. Scene coordinates remain separate from the
actual file header; GeoTIFF cannot store every coordinate.

Native Item reading retains composite `start_datetime` and `end_datetime`
coordinates. `StacSource` instead keeps ODC's time axis and grouping, without
reconstructing interval bounds or squeezing a one-Item result. Retain source
Items when their composite intervals are needed for later publication.
Changing the time axis or reducing pixels does not recompute provenance.
Drop or recompute coordinates that no longer describe the result.

Full Collection definitions belong to catalog tables, under the JSON-serializable
`stac:collections` DataFrame attr. `catalog.gs.collections` exposes native copies;
Dataset ACDD attrs contain only the selected descriptive fields for that raster.
Use `GeoCatalog.concat` to combine tables while preserving their definitions.
