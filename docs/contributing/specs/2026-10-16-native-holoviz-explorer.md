# Native HoloViz explorer product brief

Status: library direction and product requirements agreed; implementation
architecture intentionally open until a measured browser proof.

This replaces the earlier prescriptive version of this document. Start with
[the handoff](../handoffs/2026-10-16-holoviz-explorer/HANDOFF.md), which records
preservation boundaries, previous evidence, and relevant documentation.
Older HoloViz and Lonboard implementation plans are not the new architecture.

## Required outcome

Replace the interactive Lonboard explorer with a native HoloViz implementation:
Panel/Param, hvPlot/HoloViews, GeoViews, Bokeh, and Datashader where useful.
Preserve working static `.gs.plot()`, `shapes.py`, their shared helpers,
public explorer entry points, and unrelated working-tree changes.

A compact GIS sidebar contains layers, visibility, selected-layer controls,
legends and information/identification. The map sits beside it. A collapsible
catalog table sits below the map with property/date filters and row checkboxes.
Checking rows opens rasters automatically; unchecking removes them, including
under sorting/filtering/pagination. No Add raster button.

For each raster:

- Honor explicit band/RGB arguments; otherwise use declared RGB, or the first
  displayable spatial band in Dataset order if no RGB is declared.
- Let users choose a single band or assign Red, Green and Blue bands.
- Remember independent per-raster display choices without mutating source data.
- Provide meaningful continuous/categorical legends, RGB stretch and opacity.

Color by is for vector attributes. Search is optional for long option lists.
Include imagery/light/dark/OSM basemaps, responsive styling, native lazy xarray
behavior, and efficient COG reads with correct metadata/nodata handling.
The same explorer must be usable in VS Code notebooks, JupyterLab, Colab and
serving; report each environment's actual validation status.

## Proof before architecture

The first deliverable is a small runnable native UI with a selectable table,
vector, COG and computed lazy xarray raster. Demonstrate browser rendering,
checkbox opening/removal, band/RGB selection, pan/zoom and a style change.
Measure latency and source/Dask work, including repeated interactions.
No eager whole-raster computation or unmeasured claims of efficient reads.

Use supported dependency capabilities first. Decide code structure after the
proof. No required layer classes, DynamicMap arrangement, persistent-pane
strategy, cache, registry, separate rendering pipelines or widget lifecycle
scheme. Address demonstrated library constraints with the simplest supported
solution rather than restoring old abstractions.

The first proof need not implement every former feature. Inventory existing
public behavior before the full replacement. Preserve meaningful contracts;
replace obsolete implementation-specific tests. Static plot polish and the
missing copy action remain separate follow-ups.
