# Catalogs, observations, and windows

A catalog table describes observations and the files holding their pixels.
A chip is a pixel window of a prepared raster. These are different stages:

```text
Item rows → select observations → open assets → choose a grid → raster chips
```

Start with [catalog creation and loading](../guides/catalogs-and-cuts.md) for
Items, or [chip reading and merging](../guides/chips.md) for raster windows.

## What an Item row represents

An Item has an identity, geometry, an observation time or period, properties,
and assets. Each asset has an `href`; its pixels stay in the file. Collection
definitions live in the table's `stac-geoparquet` attrs and are exposed as
`rows.gs.collections`. Filtering rows retains those definitions.

A standalone Series does not inherit the frame's CRS. Use a one-row
GeoDataFrame when CRS context matters. STAC Item coordinates are WGS84;
conversion from a GeoDataFrame can reproject them.

## Native grids and a combined Dataset

`rows.gs.to_datatree()` returns `collection/Item/asset` nodes. Each asset holds
its own grid, variables and band metadata. Item time and properties are scalar
coordinates on the Item node. Asset Datasets obtain them with
`.to_dataset(inherit="all_coords")`.

`tree["collection"].gs.combine()` merges each Item's assets and stacks Items
along `time`. It requires a shared CRS, resolution and pixel alignment, with
compatible pixel meanings for each stacked variable. It spans the union of
extents and fills absent pixels or assets with NaN. It does not mosaic Items
into a single date. Use `stac.load_dataset` for grouped mosaics on a chosen grid.

Grid selection and resampling are explicit. A CRS alone reprojects each asset
independently; an exact GeoBox places all assets on the same output grid.
Keep source trees open until their dependent calculations finish.

## Temporal windows and sampling status

`GeoSamples` and `rows.gs.sample()` are unavailable while sampling is redesigned.
There is no supported calendar-window grouping contract at present. Earlier
options such as `freq`, `window`, and `interval` are not a current API.

For a prepared Dataset, use native xarray `.sel` or `.isel` to choose dates
before constructing Chips. These select time labels; interval overlap needs an
explicit mask against `start_datetime` and `end_datetime`. Use
[temporal composites](../reference/rasters.md#composites-and-time-selection)
when reducing observations into calendar bins.

For forecasting, select available history before interpolation or reduction
that could use future observations. A pixel window alone does not establish
an observation's availability or a train/validation split.

## Portability and resource ownership

Saved Item tables use relative asset hrefs. Move the table together with the
asset folder and retain their relative layout. Opened rows remember the table's
source location, so `.gs.to_items()` can resolve those addresses.

`concat_catalogs` merges compatible Collection definitions. It currently
refuses tables read from different files, whose hrefs resolve against different
sources. Plain pandas concatenation can discard differing metadata.

Closing a loaded tree closes its files. Chips and combined arrays do not take
ownership of source resources; consume lazy pixels before closing those sources.
