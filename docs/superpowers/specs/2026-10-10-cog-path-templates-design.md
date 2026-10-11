# COG path templates

Date: 2026-10-10. Supersedes the `to_cog` part of
`2026-10-06-name-format-design.md`.

## Goal

`to_cog` takes one path that is also a `str.format` template. The fields it
names decide how a raster is split into files. The `name` argument, the
`split_bands` flag and the fixed `<name>/<name>_<instant>/` layout go away.

## Caller

`image` is one tile with four bands and one date; `months` is the same place
on three dates.

```python
image.gs.to_cog("out/tile.tif")
# out/tile.tif                                 all four bands

image.gs.to_cog("out/s2_{band}.tif")
# out/s2_B02.tif ... out/s2_B08.tif            one file per band

image.gs.to_cog("out/{prefix}_{band}.tif", prefix="s2")
# out/s2_B02.tif ... out/s2_B08.tif

months.gs.to_cog(
    "out/{scene}.tif",
    scene=lambda raster: raster.attrs["stac_items"][0]["id"],
)
# one file per date, each named by its own Sentinel-2 scene id

months.gs.to_cog("out/tile_{time:%Y%m%d}.tif")
# out/tile_20190130.tif, out/tile_20190301.tif, out/tile_20190402.tif

months.gs.to_cog("out/{time:%Y%m%d}/{band}.tif")
# 3 dates x 4 bands = 12 files

image.gs.to_cog("out/{lat:.3f}_{lon:.3f}_{time:%Y%m%d}.tif")
# out/19.921_-100.291_20190130.tif

months.gs.to_cog("out/tile.tif")
# ValueError: 3 dates would share out/tile.tif; add {time} to the path,
# or save with to_zarr or to_netcdf, which hold dates in one store

sample.gs.to_cog("out/{site}/{group}.tif", site="dw_001")  # a stack
# out/dw_001/image.tif, out/dw_001/label.tif
```

## Fields

| Field | Value | Shapes the layout |
| --- | --- | --- |
| `time` | this file's instant, a `datetime` | yes: differs per date |
| `band` | this file's variable name | yes: differs per variable |
| `group` | the stack group, stacks only | yes: differs per group |
| `lat`, `lon`, `res`, `width`, `height`, `epsg`, `start`, `end` | from `GeoAnchor.format`, unchanged | no: one value per raster |
| any keyword the caller passes | a value, or a callable given the file's raster | only a callable can |

Any keyword name works. A plain value is filled in as given. A callable is
called once per file with that file's raster (one date, and one band where
`{band}` splits) and its result filled in, so it can differ per date:
`value(raster) if callable(value) else value`. A keyword that repeats a
built-in field name raises `TypeError`.

## Layout rule

The writer renders the template for every (date, band) pair and groups the
pairs by the path they render to. No template is parsed.

```python
files: dict[str, list[tuple[Timestamp | None, str]]] = {}
for time in dates:                       # [None] for an undated raster
    for band in raster.data_vars:
        part = raster_for(time, band)
        values = {key: value(part) if callable(value) else value for key, value in extra.items()}
        path = anchor.format(template, time=time, band=band, **values)
        files.setdefault(path, []).append((time, band))
```

- Pairs that render to one path share one file, so bands without `{band}`
  stay one multi-band COG.
- A path shared by two dates raises `ValueError`, naming the path and the
  `{time}` field. This also catches partial dates, as `{time:%Y}` over two
  dates of one year.
Errors name the path or field and show a corrected example:

| Mistake | Message |
| --- | --- |
| several dates, no `{time}` | `3 dates would share out/tile.tif; add {time} to the path, as out/tile_{time:%Y%m%d}.tif, or save with to_zarr or to_netcdf` |
| partial date collides | `2019-01-30 and 2019-03-01 would share out/2019.tif; make {time:%Y} more precise, as {time:%Y%m%d}` |
| `{time}` on an undated raster | `this raster has no dates, so {time} cannot be filled; drop it from the path` |
| keyword repeats a field | `band= repeats the field {band}, which the writer fills; name it something else` |
| path is not a TIFF | `to_cog writes GeoTIFFs; end the path in .tif` |

Any other unknown field raises Python's own `KeyError`.

## Reading back

Nothing parses names. Every file GeoSave writes stores its band names and date
in its header, so `open_raster` on a folder rebuilds bands and dates whatever
the files are called. `create_items` already groups files into Items by the
instant in their headers, so it needs no change beyond the paths it is given.

Deferred: a single-band file with no stored band name, as AWS-style
`<scene>/B04.tif`, reads as `band_1`. A fallback taking the file stem as the
band name would let `open_raster` read such folders. It is a reader change of
its own and is not part of this spec.

## Changes

| File | Change |
| --- | --- |
| `geodata/core/anchor.py` | `format(template, **extra)`; `extra` fills further fields; a repeated field raises `TypeError` |
| `geodata/io/raster/cogs.py` | `layout(ds, template, **extra)` implements the rule above; `write` takes the template; `split_bands` and `CogFile.scene` go |
| `geodata/core/raster.py` | `to_cog(path, *, overwrite, storage_options, map_scale, **options)`; keywords that are not COG options fill fields |
| `geodata/core/stack.py` | `to_cog` adds the `group` field; one call writes every group |
| `geodata/core/array.py` | `to_cog` follows the raster's, without `{band}` |
| `workflow/tasks/sample.py` | drop `split_bands` from its publication options |
| docstrings of all `to_cog` | list every field with an example, the layout rule, and the errors |
| `examples/02_rasters.ipynb` | `to_cog("predicted.tif")` |

Keywords: COG creation options and template fields share `**`. The COG
options are a closed `TypedDict`; any other keyword fills a field. A keyword
that is both raises `TypeError`.

## Tests

`tests/geodata/io/raster/test_cogs.py`, `tests/geodata/core/test_raster.py`,
`tests/geodata/core/test_stack.py`, `tests/geodata/core/test_anchor.py`:

- a template without fields writes one multi-band file;
- `{band}` writes one file per band, and the bands round-trip by name;
- `{time}` writes one file per date, and `open_raster` on the folder restores
  the dates;
- `{time}` and `{band}` together write dates x bands files;
- a partial date (`{time:%Y}`) over two dates of one year raises;
- several dates without `{time}` raise, naming the path;
- `{time}` on an undated raster raises, with the message above;
- a caller keyword fills a field; one repeating a field raises `TypeError`;
- a callable keyword is called once per file with that file's raster, and
  can give each date its own name;
- a stack's `{group}` writes one file per group, and `create_stack_items`
  builds the same Items as before from those files;
- a rendered path without a TIFF suffix raises;
- existing callers in `tests/` move to templates; every `split_bands=True`
  becomes a `{band}` template.

## Out of scope

- Templates for `to_zarr` and `to_netcdf`; they hold dates and bands in one
  store and take a plain path.
- Parsing names in `open_raster` (see "Reading back").
- More fields (`x`, `y`, tiling schemes, place names), added when a caller
  needs one.

## As built (2026-10-10)

Departures from the text above, decided while implementing:

- A callable field is given each date's whole raster, every band, not one
  band. Called per band, a callable could split bands without `{band}`
  being named.
- A stack's `to_cog` also takes group names mapped to templates, for a stack
  whose groups differ in dates (a dated image beside an undated elevation
  model); `{time}` on an undated group says so. Every group is planned before
  any file is written, and two groups sharing a path raise.
- An unknown field raises `GeoAnchor.format`'s existing `ValueError`, which
  lists the fields, rather than `KeyError`.
- At the `io` level, `cogs.write` and `cogs.layout` take fields as an explicit
  `fields=` mapping; only the accessors split `**keywords` into COG creation
  options and fields.
- `Array.to_cog` writes through its one-band raster, so it now takes templates
  and returns a tuple, and a dated band no longer refuses.
- Built-in fields are only `time`, `band` and `group`, the ones that can split
  files (user decision, 2026-10-10). The anchor fields (`lat`, `lon`, ...) are
  not filled by the writer; a caller passes them, usually as a callable such
  as `lat=lambda raster: raster.gs.anchor.lonlat[1]`. `GeoAnchor.format` is
  unchanged, and `create_items` keeps it for Item ids. An unfilled field
  raises Python's `KeyError`.
