# Metadata rules

Read a validated metadata value from an ordinary xarray band:

```python
import numpy as np
import xarray as xr

import geosave as gs
from geosave.geodata import attrs

band = xr.DataArray(np.array([[0, 1], [2, 255]], dtype="uint8"), dims=("y", "x"))
band = band.gs.write_nodata(255).gs.write_flags(
    {0: "background", 1: "forest", 2: "water"}
)
attrs.CFFlags.from_attrs(band.attrs)
```

Output:

```text
CFFlags(flag_values=[0, 1, 2], flag_masks=None, flag_meanings='background forest water')
```

The attrs remain a plain xarray mapping. A group reads a few related keys as
one validated value; GeoSave keeps no second metadata store. Reading an absent
group returns `None`. Reading invalid metadata raises instead of guessing.
Use the `.gs` properties and `write_*` methods for everyday work, as shown in
[describe raster values](../basics/attrs.md).

## What the groups mean

| Group | Keys | Meaning |
| --- | --- | --- |
| `CFPacking` | `scale_factor`, `add_offset`, `_FillValue`, `missing_value`, `valid_min`, `valid_max`, `valid_range`; also reads `nodata` | CF packed data (§8.1) and missing data (§2.5.1) on each variable; `xr.decode_cf` reads them together. |
| `CFDescription` | `long_name`, `standard_name`, `units` | What a variable's values are (CF chapter 3); `band.gs.description`. |
| `CFFlags` | `flag_values`, `flag_masks`, `flag_meanings` | Class codes or bit flags and their labels (CF §3.5); `band.gs.flags`. |
| `EOBand` | `common_name`, `center_wavelength`, `full_width_half_max` | Band identity and wavelength in micrometres, as STAC `eo` states them. |
| `Display` | `rgb` (Dataset), `color_map` (variable) | How a raster draws; the GeoTIFF writer maps it to GDAL colour interpretations and colour tables. |
| `ACDD` | `id`, `title`, `summary`, `keywords`, `institution`, `license` | Dataset identity and discovery metadata on root attrs. |
| `ACDDExtent` | `geospatial_bounds`, `geospatial_bounds_crs` | Declared spatial extent as WKT and its CRS, retained on raster roots and spatial variables. |
| `StacCollection` | `stac_collection`, `stac_extensions` | The STAC Collection and extension URIs on root attrs. |

Band groups live on variables. A time label's cell is not an attr group: it is
two coordinates on the time dimension. Zarr stores them as they are; netCDF
stores CF bounds:

```text
in memory                              netCDF
start_datetime(time) ┐   encode  ──►   time:bounds = "time_bounds"
end_datetime(time)   ┘   ◄── decode    time_bounds(time, bounds)
```
Other keys, such as `units` and `long_name`, stay ordinary attrs. ACDD keys
also stay in that mapping; the group reads their dataset-level meaning.

Write a group with `attrs.stamp_attrs(obj, value)` and remove one with
`attrs.clear_attrs(obj, Group)`. Both return a shallow copy of the object,
so the original keeps its attrs. Construct a new group to replace a declaration.

```python
band = attrs.stamp_attrs(band, attrs.CFPacking(fill_value=255))
band = attrs.clear_attrs(band, attrs.CFPacking)
band = attrs.stamp_attrs(band, attrs.Display(color_map={0: "#000000"}))
```

Writing a group removes all its previous keys before stating the new value.
This avoids stale aliases. `write_nodata` casts the fill to the band's dtype;
`write_flags` checks class codes against it. Generic
`stamp_attrs` only replaces the group. Writes share pixels and do not compute
Dask arrays or recode classes.

GeoSave's `gs.to_array()` requires compatible active CF declarations
before stacking bands, including units, unsigned interpretation, flags, validity
ranges, calendars and cell methods. Missing units do not agree with declared
units. `combine_attrs` cannot override or discard these declarations.
Native `xr.concat` does not check packing or CRS:
decode and align inputs explicitly before combining them. Attr combination
follows xarray's policies and does not validate pixel semantics.

## Decode and encode variable values

Stored DN carries scale, offset and missing-value declarations in variable
attrs. `raster.gs.mask_and_scale()` uses xarray's CF decoder and retains the
consumed fields and original dtype in each variable's encoding.
Decoding preserves lazy arrays and leaves the source unchanged. Writers encode
physical values from their native encoding; raw DN is written with its CF attrs.
There is no separate public reverse-conversion step.

The DataArray accessor decodes one band's values and retains its coordinates.
The Dataset accessor delegates to `xr.decode_cf(..., decode_coords="all")`,
including coordinate packing, time bounds and CF relationships. Decode a
Dataset before selecting a band when sibling bounds are needed. DataTree has
no decoding accessor; select the Dataset needed for analysis first.
Native CF names and numeric declarations are the input contract.

The STAC loader declares ODC's output nodata in the loaded dtype using
rioxarray, so the CF fill matches the actual pixels. For a manually constructed
array, use `.gs.write_nodata(value)` to declare `_FillValue` at the array's
precision through native rioxarray. Direct attrs assignments follow xarray's
behavior, including its warnings and errors; the decoder no longer converts
textual scaling or casts mismatched missing-value declarations.


`write_nodata()` and `write_packing()` only declare what existing pixels
mean; neither changes pixels. Each preserves the other's declarations.

An undecoded read (`mask_and_scale=False`) retains DN with CF attrs. A decoded
read retains the storage recipe in encoding. Writers save either representation
without applying scaling twice. STAC loading retains DN by default and translates published band metadata into
attrs. Use `load_dataset(..., mask_and_scale=True)` to decode immediately, or
call `.gs.mask_and_scale()` on the loaded Dataset afterward. Before loading,
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

Retained GeoSave combinations accept native policy strings or callables and
warn about omitted keys. In `gs.to_array()` this policy applies only to
descriptive attrs; validated CF pixel declarations are always retained. Their default is `"drop_conflicts"`. Native `xr.concat`
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

NetCDF and Zarr writers use native xarray encoding. The Dataset GeoTIFF writer
encodes each raster variable with xarray's CF codec before rioxarray stacks the
bands, preserving per-band scale and offset. All use native quantization and
casting behavior; GeoSave adds no integer overflow or fill-collision preflight.
Choose an appropriate dtype, scale and fill, and verify output with a round trip.

GeoTIFF translates these groups into native band properties and tags. STAC
Items describe them through Raster, EO, and Classification metadata. A STAC
load restores the selected metadata that describes the loaded bands;
xarray attrs are not a lossless copy of the full Items or Collection.

Zarr records variable order in a storage attr and restores it on opening.
Metadata support varies by format, so verify a round trip for the properties
your workflow needs. See the [attrs reference](../reference/attrs.md) and
[read and write guide](../basics/io.md).

A raster loaded from STAC carries its STAC metadata in two named places.
The root `StacCollection` attrs hold the Collection and the Items' extension
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
