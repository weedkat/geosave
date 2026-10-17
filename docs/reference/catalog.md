# Catalogs

An item table holds one STAC Item per row. `items.gs.sample()` groups equal
spatial footprints into gridless rows, retaining all dates and native grids.
`samples.gs.frame(windows={"image": {"slots": ["0D"]}})` selects temporal bins within
each sample; `frames.gs.chip(128, target=grid)` records explicit pixel windows.
An Item row opens as a Dataset; a sample row opens as a DataTree, both through
`row.gs.to_xarray()`. See [the catalog guide](../guides/catalogs-and-cuts.md).

## Item tables

::: geosave.geodata.core.vector.catalog.GeoCatalog

## Sample tables

Use `GeoSamples.concat([first, second])` to combine sample tables and preserve
Collection definitions. When a Collection id repeats, the first definition wins.

Sample tables retain their accessor when the active geometry column is renamed.
Framing keeps that geometry and its CRS; chipping updates the named geometry
column with each chip's WGS84 footprint. Save samples with `.gs.to_geoparquet()`
and reopen them with `gs.open_vector()`. Their GeoJSON and GeoPackage accessor
writers reject exports because those formats would lose Collection metadata.

::: geosave.geodata.core.vector.samples.GeoSamples

## Read an Item row

::: geosave.geodata.core.vector.item.GeoItem

## Read a sample row

`row.gs.geobox` and `row.gs.anchor` are None before chipping. `row.gs.region`
returns the read footprint as an ODC Geometry, excluding chip padding.
`row.gs.timespan` gives the sample's observed span or frame's output dates.

`row.gs.key` identifies the spatial sample and, for chips, its grid and padding.
Chipping stores this in `parent`; merging also matches frame dates. Old
reference-based sample tables must be regenerated from their Item catalogs.

::: geosave.geodata.core.vector.sample.GeoSample
