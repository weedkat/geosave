# Panel explorer replacement

The explorer uses Panel controls in notebooks and on a server: a compact
sidebar, Lonboard map, then a collapsible Tabulator catalog. Panel owns the
controls, tabs, table, and charts. Lonboard owns map rendering and COG tiles.
The existing domain renderers remain separate from the UI; the dock/plugin
framework is removed without aliases. Notebook display uses an approved native
ipywidgets GridBox around the Lonboard widget and Panel-rendered controls.

## Evidence before implementation

A throwaway proof in `/tmp/geosave-panel-probe` rendered a vector, tiled COG,
and dask-backed xarray raster. Chromium checkbox interactions passed under
`panel serve` and JupyterLab native comms. The original nested ipywidgets comm path failed and was replaced with a native
GridBox. Actual VS Code rendering and checkbox interactions passed; Google Colab is unverified. A local
Jupyter test of their transport does not certify those products.

## Work

- [x] Replace app/docks with one `Explorer(pn.viewable.Viewer)`, native Panel
  controls and a persistent layout; keep `.gs.explore()`, composition and
  layer settings. Remove the dock API rather than emulate it.
- [x] Let Tabulator own filtering and positional selection. Open rows through
  `row.gs.to_xarray()`; unchecking removes and closes their widgets. Surface
  opening failures without corrupting the selection or other layers.
- [x] Keep searchable color selection, legends, information, identification,
  time selection and comparison. Dispose watchers and removed widgets.
- [x] Bound xarray sample/window materialization and prove it with lazy-task
  checks. Continue using Lonboard's `from_geotiff` for matching COG grids.
- [x] Run focused behavioral tests and lint. Repeat actual browser interactions
  on the final app, record each deployment path separately, and document
  dependency limitations rather than build a private comm implementation.

## Review focus

Filtered and paginated row positions; independently owned viewer state;
failed item opens; catalog removal; lazy arrays larger than a map tile;
nodata/classes; signed/authenticated URLs; repeated notebook display.

Validation and remaining blockers: [browser audit](../audits/2026-10-16-panel-lonboard.md). The requested environment matrix is not complete.
