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
| `Packing` | `scale_factor`, `add_offset`, `_FillValue`, `missing_value`; also reads `nodata` | CF conversion and missing-value declarations on each variable. |
| `Description` | `long_name`, `standard_name`, `units` | What a variable's values are (CF chapter 3); `band.gs.description`. Find variables by them with `ds.cf.standard_names`. |
| `Legend` | `flag_values`, `flag_masks`, `flag_meanings`, `color_map` | Class codes or bit flags and their labels. |
| `Spectral` | `common_name`, `center_wavelength`, `full_width_half_max` | Band identity and wavelength in micrometres. |
| `GDALBand` | `variable_name`, `colorinterp` | Band naming and colour interpretation in GDAL formats. |
| `ACDD` | `id`, `title`, `summary`, `keywords`, `institution`, `license` | Dataset identity and discovery metadata on root attrs. |
| `Extent` | `geospatial_bounds`, `geospatial_bounds_crs` | Declared spatial extent as WKT and its CRS, retained on raster roots and spatial variables. |

Band groups live on variables. A time label's cell is not an attr group: it is
two coordinates on the time dimension, stored in files as CF bounds:

```text
in memory                              netCDF / Zarr
start_datetime(time) ┐   encode  ──►   time:bounds = "time_bounds"
end_datetime(time)   ┘   ◄── decode    time_bounds(time, bounds)
```
Other keys, such as `units` and `long_name`, stay ordinary attrs. ACDD keys
also stay in that mapping; the group reads their dataset-level meaning.

Write a group with `attrs.stamp_attrs(obj, value)` and remove one with
`attrs.clear_attrs(obj, Group)`. Both return a shallow copy of the object,
so the original keeps its attrs. Construct a new group to replace a declaration.

```python
band = attrs.stamp_attrs(band, attrs.Packing(fill_value=255))
band = attrs.clear_attrs(band, attrs.Packing)
band = attrs.stamp_attrs(band, attrs.GDALBand(colorinterp="red"))
```

Writing a group removes all its previous keys before stating the new value.
This avoids stale aliases. Concrete accessor writes such as `write_nodata`
and `write_legend` also validate values against the band's dtype; generic
`stamp_attrs` only replaces the group. Writes share pixels and do not compute
Dask arrays or recode classes.

GeoSave's `gs.to_dataarray()` requires compatible active CF declarations
before stacking bands. Native `xr.concat` does not check packing or CRS:
decode and align inputs explicitly before combining them. Attr combination
follows xarray's policies and does not validate pixel semantics.

## Decode and encode variable values

Stored DN carries scale, offset and missing-value declarations in variable
attrs. `raster.gs.mask_and_scale()` uses xarray's CF decoder and retains the
consumed fields and original dtype in each variable's encoding.
`raster.gs.scale_and_fill()` applies the reverse conversion without writing a
file, returning stored pixels with CF attrs. Both preserve lazy arrays and
leave the source unchanged. Integer encoding rejects overflow and collisions
with missing-value codes when the pixels are evaluated.

`write_nodata()` and `write_scale_offset()` only declare what existing pixels
mean; neither changes pixels. Each preserves the other's declarations.

An undecoded read (`mask_and_scale=False`) retains DN with CF attrs. A decoded
read retains the storage recipe in encoding. Writers save either representation
without applying scaling twice. The STAC loaders support the same option after
loading DN through ODC and translating published band metadata. Before loading,
GeoSave rejects selected source bands with different effective scales or offsets.
ODC may still remap different source nodata codes to one output fill value.

## Join data with consistent meaning

```python
joined = xr.concat([band, band], dim="time", combine_attrs=attrs.combine_attrs)
joined
```

`attrs.combine_attrs` is an explicit xarray callback; ordinary xarray operations
do not automatically apply it. It delegates combination to xarray and adds
`DroppedAttrsWarning` for omitted keys. Its default, `"drop_conflicts"`, drops
keys with disagreeing values and retains keys present in only one input.
Groups do not affect this key-by-key policy, and aliases are not normalized.

Retained GeoSave combinations, including `gs.to_dataarray()` and DataTree
`gs.to_dataset()`, accept native policy strings or callables and automatically
warn about omitted keys. Their default is `"drop_conflicts"`. Native `xr.concat`
and `xr.merge` default to `"override"`; pass a policy explicitly when that is
not the desired behavior. Direct native calls need `attrs.combine_attrs` to
add GeoSave warnings.

Warnings describe absent keys, not changed values: `"override"` can select a
different value without omitting its key. Explicit `"drop"` warns about every
omitted key. No alias normalization or group validation occurs during combination;
a resulting partial legend may fail validation when later read. Check physical
units and class meanings before combining pixels, independently of attrs policy.

## Decode and encode are different stages

Decoded values retain their reverse storage recipe in `encoding`. Native
arithmetic and reductions generally discard data encoding; choose a new output
encoding when saving a derived variable. `.gs.composite()` requires explicit
decoding first and does not restore the old integer recipe after reduction.

NetCDF and Zarr writers use native xarray encoding. They do not add GeoSave's
integer overflow or fill-collision checks. To request checked conversion,
configure the decoded variable's encoding and call `.gs.scale_and_fill()`
before writing the resulting stored values. GeoTIFF already uses this checked
conversion. Verify the chosen representation with a round trip.

GeoTIFF translates these groups into native band properties and tags. STAC
Items describe them through Raster, EO, and Classification metadata. A STAC
load restores the selected metadata that describes the loaded bands;
xarray attrs are not a lossless copy of the full Items or Collection.

Zarr records variable order in a storage attr and restores it on opening.
Metadata support varies by format, so verify a round trip for the properties
your workflow needs. See the [attrs reference](../reference/attrs.md) and
[read and write guide](../basics/io.md).

A raster loaded from STAC carries its STAC metadata in two named places.
The root `StacMetadata` attrs hold the Collection and the Items' extension
schemas. The `stac_properties` coordinate along `time` holds one JSON mapping
per observation: the properties every loaded Item shares, plus those requested
with `with_properties`. `gs.stac_common_properties` is their intersection.
`to_catalog` builds each Item's properties from these records; no other
coordinate becomes a property. Dates, `proj:*` and band fields always come
from the raster's own coordinates, grid and attrs.

```text
source                                   cell of the label
Item with start/end, one label per Item  the Item's start_datetime, end_datetime
groupby="solar_day" mosaic               the local solar day, e.g. [01-09 17:00, 01-10 17:00) UTC
.gs.composite(freq="1MS", ...)           the resample bin, e.g. [Jan 1, Feb 1)
.gs.write_time_bounds("MS")              one freq step per label
```

`.gs.composite()` also writes `cell_methods` and rebuilds `stac_properties`.
Other operations that change the time axis do not recompute cells or
provenance: native reductions drop the cell coordinates; restate them with
`gs.write_time_bounds` when they still apply. Publication writes the cells
back as STAC `start_datetime` and `end_datetime`.

Full Collection definitions belong to catalog tables, under the JSON-serializable
`stac-geoparquet` DataFrame attr; `catalog.gs.collections` exposes native copies.
A Dataset's `gs.stac_collection` holds the one it was loaded from. Attrs
describe one load: native concat keeps the first one's attrs by default, while
an explicit combination policy may drop conflicting metadata. Request a property
with `with_properties` to keep it per date. Use
`GeoCatalog.concat` to combine tables while preserving their definitions.
