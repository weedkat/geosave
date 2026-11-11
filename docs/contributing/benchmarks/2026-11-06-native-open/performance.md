# Native versus ODC loading performance

This measures the surviving APIs on equivalent aligned pixels. It is not a
before/after measurement of the opener rename or catalog-helper consolidation.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python \
  docs/contributing/benchmarks/2026-11-06-native-open/performance.py \
  --output docs/contributing/benchmarks/2026-11-06-native-open/performance.json

MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python \
  docs/contributing/benchmarks/2026-11-06-native-open/performance.py \
  --size 2048 \
  --output docs/contributing/benchmarks/2026-11-06-native-open/performance-2048.json
```

## Method

- Local tiled DEFLATE GeoTIFFs; three separate uint16 assets per Item,
  deterministic random pixels, eight different dates, one Collection.
- Compare one Item and eight Items, at 1024² and 2048² pixels per band.
- Native: `io.open_item` for one Item, `rows.gs.to_datatree()` for eight.
  The catalog table is built before timing; its conversion back to Items is
  included in native setup.
- ODC: `io.load_dataset` and `io.load_datatree`, explicit exact GeoBox,
  `groupby="id"`, uint16 and matching nodata. No resampling or mosaicking.
- Same 512×512 spatial chunks; Dask threaded scheduler with four workers.
- One untimed warmup per method, five measured repetitions, rotated method
  order. Fixtures, imports and correctness checks are outside timings.
- Warm filesystem caches; caches are not cleared. Not a cold-start or remote
  network benchmark.
- Separate setup, compute and close timings. Each sample reopens the sources.
  Setup returns lazy arrays; a separate untimed task probe confirms zero Dask
  tasks during setup and counts executed tasks during computation.
- Every measured result is checked for exact grid, dtype and every pixel.
  Native single-Item scalar time is normalized only for comparing pixel values.
- Reported totals are medians of sample totals, so they need not equal the sum
  of independently reported component medians.
- Raw JSON includes samples, dependency versions and platform. Memory use and
  actual physical disk reads are not measured.

## Results

Median seconds across five measured repetitions.

| Size | Items | Method | Setup | Compute | Total | Total range | Tasks |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 1024² | 1 | native | 0.060 | 0.041 | 0.099 | 0.087–0.110 | 27 |
| 1024² | 1 | odc_dataset | 0.014 | 0.044 | 0.058 | 0.052–0.062 | 18 |
| 1024² | 1 | odc_datatree | 0.019 | 0.043 | 0.064 | 0.060–0.068 | 18 |
| 1024² | 8 | native | 0.571 | 0.253 | 0.837 | 0.787–0.881 | 195 |
| 1024² | 8 | odc_dataset | 0.027 | 0.343 | 0.370 | 0.345–0.446 | 123 |
| 1024² | 8 | odc_datatree | 0.034 | 0.354 | 0.413 | 0.345–0.462 | 123 |
| 2048² | 1 | native | 0.055 | 0.144 | 0.200 | 0.187–0.216 | 99 |
| 2048² | 1 | odc_dataset | 0.017 | 0.162 | 0.179 | 0.167–0.202 | 54 |
| 2048² | 1 | odc_datatree | 0.018 | 0.162 | 0.188 | 0.172–0.192 | 54 |
| 2048² | 8 | native | 0.512 | 1.096 | 1.667 | 1.373–1.754 | 771 |
| 2048² | 8 | odc_dataset | 0.042 | 1.395 | 1.440 | 1.270–1.790 | 411 |
| 2048² | 8 | odc_datatree | 0.060 | 1.472 | 1.517 | 1.364–1.851 | 411 |

## Interpretation

ODC Dataset loading was approximately **2.26× faster overall** for eight
1024² Items (0.370 versus 0.837 s), and **1.16× faster overall** for eight
2048² Items (1.440 versus 1.667 s). ODC substantially reduced setup cost.
Native pixel computation was faster in both cases; the end-to-end advantage
narrowed as pixel volume increased. Fewer Dask tasks did not imply faster
pixel computation in these runs.

For these aligned single-grid inputs, ODC DataTree loading added some overhead
over ODC Dataset loading. This does not measure mixed-resolution DataTrees.
The lower setup time is consistent with ODC metadata-first planning, but this
benchmark does not profile individual functions or prove a specific bottleneck.

Use ODC for explicit multi-Item grid assembly. These results do not justify
replacing native catalog semantics: the companion feasibility study demonstrates
grid, temporal and dtype differences. Remote I/O, cold caches, reprojection,
very large Item counts and memory consumption need separate benchmarks.

All 60 measured operations (12 cases × five repeats), 12 warmups and 12
separate task probes completed. Exact pixel/grid/dtype checks passed for every
measured operation; lazy setup checks passed. Targeted Ruff and whitespace
checks passed. No production code was changed.
