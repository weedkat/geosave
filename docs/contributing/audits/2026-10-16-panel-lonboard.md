# Panel–Lonboard explorer validation

Status: **cross-environment validation incomplete** (2026-10-16).

The explorer uses native Panel controls: compact layer controls, a
Lonboard map, and a collapsible Tabulator table below the map. Catalog row
selection calls the public row reader automatically. Clearing selection
removes and releases the corresponding raster widgets. Property filters
belong to Tabulator; catalog date filtering preserves interval overlap.
There is no Add raster button, custom dock system, or custom widget communication
implementation. Notebooks use a standard ipywidgets GridBox: Lonboard remains
a native widget, and `pn.ipywidget` renders the Panel sidebar and table. The
served app uses Panel FlexBox and its IPyWidget pane.

The xarray/vector rendering code was retained and corrected where needed.
This is a replacement of the explorer UI architecture, not a rewrite of the
geospatial I/O stack. Static `.gs.plot()` remains separate.

## Repeatable proof

```bash
uv run panel serve scripts/smoke_explorer.py
```

The proof creates a local COG, a STAC item table referencing it, one vector
with categorical and numeric properties, and one dask-backed xarray raster.
Fixture writing is explicit; the explorer reads tiles lazily. In a notebook:

```python
import panel as pn
pn.extension("tabulator")
from scripts.smoke_explorer import create_explorer
create_explorer()
```

`tests/geodata/viz/explorer/test_browser.py` repeats the served proof using
Playwright. Install Playwright and its browser in the test environment, then:

```bash
uv run pytest -m slow tests/geodata/viz/explorer/test_browser.py
```

`GEOSAVE_BROWSER` optionally names an existing Chromium executable. Browser
tests are optional slow tests, so normal library installation does not add
Playwright.

## Environment evidence

Panel 1.9.4, Lonboard 0.17.0, Python 3.14.8, local Linux Chromium/Brave.
VS Code 1.140.0 was driven through Playwright Electron in an isolated profile.
JupyterLab 4.6.4 ran with the project's Jupyter widget extensions installed
in the frontend server and the GeoSave environment as its kernel.

| Path | Evidence | Result |
| --- | --- | --- |
| Local `panel serve` | Actual browser rendering, catalog check/uncheck, COG and lazy xarray display | Passed |
| JupyterLab native comms | Actual browser execution in a notebook, catalog check/uncheck, map and table below it | Passed |
| Original nested Panel widget layout | Stayed at `Loading widget...`; replaced with the native GridBox arrangement | Failed, replaced |
| Actual VS Code 1.140.0 notebooks | Isolated Electron renderer: map, controls and catalog rendered; checking COG opened a layer and unchecking removed it | Passed |
| Actual Google Colab | No authenticated Colab runtime/browser test available in this run | Unverified |
| Public deployment behind a reverse proxy | Local server only | Unverified |

The original nested notebook layout was broken in VS Code. Rendering Lonboard
inside Panel imported a bridge that replaced `comm.create_comm` with `TempComm`;
the outer notebook output consequently had no live Jupyter comm. The approved
GridBox arrangement avoids that nesting and uses Panel's `pn.ipywidget` API for
the surrounding controls. Sidebar and table updates run in their owning Bokeh
documents to avoid stylesheet ownership errors. Restart kernels that have
already imported the broken bridge before testing the changed code.

The final GridBox proof passed real JupyterLab browser check/uncheck interactions
with a COG, vector, and lazy xarray raster. Actual VS Code rendering was observed;
catalog check/uncheck also passed in the actual VS Code Electron renderer. Colab remains unverified.

References:
[Panel notebook communication](https://panel.holoviz.org/how_to/notebook/notebook.html),
[Panel's intended AnyWidget integration](https://panel.holoviz.org/how_to/migrate/anywidget/index.html),
[Lonboard COG rendering](https://developmentseed.org/lonboard/latest/blog/posts/initial-cog/).

## Data reads and limits

- Matching tiled GeoTIFF selections use Lonboard `RasterLayer.from_geotiff`
  and async-geotiff's own tiles and overviews. Stretch reads are bounded to
  1024 by 1024 samples even when an overview is absent.
- Computed xarray rasters use spatial windows, sampled to at most 512 by 512
  before materialization and reprojection to a 256 by 256 map tile. Creation
  of a tile widget executes no dask tasks. The original raster stays lazy.
- A dask task may read its entire source chunk. Derived arrays and sources
  without overviews cannot promise COG-style range-read efficiency.
- Multi-file RGB selections, cropped/decoded arrays, signed URLs and
  authenticated remote stores were not benchmarked in this run. Multi-file
  RGB still takes the xarray window route; this remains a performance risk.
- The generic xarray tile adapter uses Lonboard 0.17's underscored raster
  constructor callbacks because that version has no public xarray factory.
  COGs use its public factory. The existing `<0.18` dependency bound remains.
- Basemap entries include Esri World Imagery, Carto Light/Dark/Streets and OSM;
  Esri/OSM attribution is supplied to the map. External provider availability
  was not established by the local fixture proof.

## Intentional API removal

Removed the explorer's dock/plugin API and `run_action` dispatcher. Use
`Explorer.add()`, `remove()`, `zoom()` and native Panel layouts. Existing
`.gs.explore()`, styled raster/vector layers, composition, time selection,
comparison, information and identify remain. The table is exposed as
`Explorer.table`. Composition copies settings so viewers do not mutate each
other. The unused direct `panel-material-ui` dependency was removed.

## Checks run

- 139 visualization tests passed after the notebook fix (one slow
  browser test excluded from that run). The earlier 287-test accessor run predates
  this notebook fix. Existing dependency/runtime warnings
  remain; this was not a repository-wide test run.
- The separate Chromium browser test passed: COG check/uncheck, searchable
  numeric Color by and legend, theme switching, and narrow-layout captures.
- Ruff and BasedPyright passed for the explorer and checked-in proof script.
- The final JupyterLab browser selection workflow passed in a fresh kernel;
  its frontend also logged an unrelated `No active debugger session` error.
- Actual VS Code notebook rendering and COG check/uncheck passed in an isolated
  editor profile. No complete cross-environment UI validation has passed: Colab
  and hosted reverse-proxy deployments remain unverified.

Server lifecycle: Panel owns document teardown. `Explorer.close()` explicitly
releases widgets while the viewer is live (for example, in a notebook).
Calling it from `on_session_destroyed` is unsafe in the installed bridge:
Lonboard's callback removal sends a widget trait update to an already destroyed
Bokeh document. That custom callback was removed instead of patching the comm
implementation. Long-running server memory reclamation is unprofiled.
