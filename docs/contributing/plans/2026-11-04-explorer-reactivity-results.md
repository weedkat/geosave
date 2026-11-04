# Explorer reactivity execution report

Plan: [2026-11-03 Explorer reactivity](2026-11-03-explorer-reactivity.md).

## Decisions and ownership

- Work stays in the shared checkout because the starting application spans existing uncommitted refactors. A read-only pre-change snapshot was saved under `/tmp/geosave-reactivity-before`; no unrelated changes were discarded or committed.
- Explorer keeps public temporal parameters and owns membership and session copies. Public actions are grouped before internal coordination methods.
- Keep `pn.ui.Page` and persistent child Viewers. LayerDetails exposes decorated views, supplies existing controls, and disconnects Param references under `param.discard_events` on release. LayerList and AttributeTables retain their layouts and reconcile membership separately from selection.
- Each layer declares rendering, legend, and summary dependencies. Map owns rendering subscriptions and a native HoloViews Counter; the counter is necessary for nested plot color-mapper invalidation. Period changes batch layer updates before one redraw. The global `layer_setting_changed` event is removed.
- Comparison state lives in `explore/comparison.py`; its controls live in `page/comparison.py`. Server registry and start/stop mechanics live in `server.py`; `.serve()`, `.stop()`, and `.close()` remain on Explorer.
- RasterLayer has a 128 MiB cache budget for reusable viewport pixel windows. Current pixels remain reusable even above that budget; rendered arrays and overview buffers are additional memory. Band/time/axis changes invalidate history. Removal and close clear buffers. Pixel reading remains in `read_scene_pixels` and the Dataset reader.
- Keep existing half-view padding and output resolution. Reducing padding or capping output area changes pan hit rate or sharpness; no unmeasured quality tradeoff was adopted.
- Retry uses the existing raster checkbox after failure, with explanatory text and a header tooltip, rather than adding another load mechanism. Fit visible layers respects date/comparison eligibility.
- Do not add background viewport workers in this iteration. The new cache addresses repeated reads without a second asynchronous lifecycle. Uncached reads remain synchronous; responsiveness during large uncached computed reads is a remaining limitation, not a claimed improvement.

## Validation ledger

- Pre-change non-slow visualization baseline: 110 passed, 8 deselected, 3 dependency warnings.
- New lifecycle/vector regressions reproduced failures before implementation and passed after it.
- New zoom-return and eviction regressions reproduced failures before cache implementation and passed after it.
- Review caught initial date synchronization and oversized-current-window reuse defects. Both were corrected; regression tests cover them.
- Single-band contrast regression exposed the need for distinct HoloViews redraw event identities; the Counter correction passes the existing color-mapper test.
- Integrated visualization run: 115 passed, 8 deselected, 3 dependency warnings before the final UI guidance changes.
- Slow suite initially: 7 passed, 1 failed (`test_native_notebook_comm`). Root cause: Material UI cannot render an empty discrete slider, even when hidden. A `No dates` placeholder fixes rendering while Param still updates options when dates become available. The notebook test then passed separately.
- Scene-pixel reader contracts: 6 passed, 20 deselected; one dependency deprecation and the existing overview/file-grid warning.
- Browser dependencies are ephemeral (`uv run --with playwright`); project dependencies were not changed.

Final checks after all production changes:

- `uv run --no-sync pytest tests/geodata/viz -m 'not slow' -q`: **115 passed, 8 deselected**, 3 dependency warnings.
- `uv run --no-sync --with playwright python -m pytest tests/geodata/viz/explore/test_server.py -m slow -q`: **8 passed**, 1 dependency warning. Includes actual Chromium raster rendering, independent footprint/raster toggles, server replacement, and notebook comms. No slow tests remain skipped in this run.
- `uv run --no-sync pytest tests/geodata/io/test_stac.py -k read_pixels -q`: **6 passed, 20 deselected**, 2 warnings described above.
- Ruff passes for the affected visualization source/tests and measurement scripts; whitespace checks pass. These are focused checks, not a claim about the entire repository suite.

## Performance evidence

[Aggregated before/after measurements](explorer-reactivity-baseline/comparison.json) use five fresh processes per version and 20 warm repetitions per ordinary interaction. The original implementation was loaded from the read-only snapshot, using the same script, dependencies, synthetic RGB input, and full Bokeh document. Single cold zoom actions have only five samples. Timings include instrumentation and use a server-side fallback viewport; they are not browser latency.

| Warm action | Before median / p95 (ms) | After median / p95 (ms) |
|---|---:|---:|
| Opacity | 63.49 / 76.24 | 62.63 / 77.24 |
| Vector color | 71.40 / 95.48 | 72.14 / 89.27 |
| Active selection | 56.90 / 62.68 | 52.79 / 72.42 |
| Cached pan | 57.18 / 63.55 | 61.21 / 76.03 |
| Visible zoom round trip | 1619.79 / 2864.88 | 1078.56 / 2158.97 |

The zoom round-trip median is about 33% lower; its p95 is about 25% lower. Other timings do not establish a broad speed improvement: cached pans are modestly slower in this measurement, and runs vary. The dependable improvements are work counts:

- Opacity: menu builds **9 → 0**, legend formats **2 → 0**, period updates **1 → 0**. Still one map draw and no source work/image-array updates.
- Active selection: menu builds **9 → 0**, no map redraw or source work.
- Vector color: only its own legend updates; no period scan or menu rebuild.
- Visible zoom round trip: source reads **2 → 0**, Dask tasks **302 → 0**. Two image-array updates remain, totaling about 44.5 MB of raw arrays in this fallback-viewport fixture.
- Cached pans: zero source reads, Dask tasks, or image-array updates in both versions.

[Budget comparison](explorer-reactivity-baseline/budgets.json): one observed window (66,816,320 bytes) causes two reads per round trip; two and three windows cause zero reads in all five repetitions. The 128 MiB default fits two such windows. Three provide no additional benefit for this two-zoom workload. This is not a universal optimal budget or a bound on total memory. Process peak RSS was roughly 3 GiB in these instrumented tests, including dependencies, transient reprojection/rendering arrays, and Bokeh serialization; reducing pixel reads does not eliminate those costs.

[4096×4096 native COG/computed benchmark](explorer-reactivity-baseline/cog.json): returning to the original COG window made zero HTTP requests and took approximately 0.4 ms in the reader-only measurement; returning to the computed view ran zero source chunks/tasks. Cold COG overview and zoom requests did perform HTTP range reads. The benchmark retains the known overview/file-grid warning from the reader. Its timings were collected during verification work and are illustrative rather than a controlled latency comparison.

[Chromium measurements](explorer-reactivity-baseline/browser.json): two catalog load cycles displayed colored raster pixels after approximately 1.23 s and 1.03 s, receiving about 9.19 MB of websocket payload each. These include metadata loading, rendering and serialization. Payload sizes exclude protocol framing/transport compression; this is not a before/after browser speed claim. The same test unchecks footprints and confirms the raster stays rendered before removing it through its independent control.

Reproduce with `scripts/measure_explorer_reactivity.py` (20 repetitions by default), `scripts/measure_explorer.py` (`GEOSAVE_PROOF_SIZE=4096`), and the slow browser test with `--junitxml` to retain its measurements. `GEOSAVE_CACHE_BYTES` selects the diagnostic cache budget.

## Remaining conditional work

- Background viewport reads and a manual-refresh mode remain deferred. The benchmark shows both source computation and image conversion/transfer costs; moving reads to workers alone would not eliminate the latter. Large uncached computed reads can still block the document thread.
- Padding/area-cap experiments and a complete browser pan/zoom latency matrix remain future performance work. Existing padding and resolution are preserved.
- The five-process timing comparison covers the computed full-page fixture; the native COG benchmark is a separate source-work measurement, not the full five-process matrix proposed in the original plan.
- Repeated raster styling still constructs HoloViews options/hover configuration on map draws. A separately measured rendered-style cache is a potential next optimization; it was not added speculatively.


## Compatibility

`viewer.period`, `viewer.freq`, and serving entry points are unchanged. The internal global change event was removed. The server registry moved from `Explorer.servers` to `explore.server.servers`; Comparison is now a Parameterized state object, with its UI built separately. No compatibility aliases were added.
