# Lazy Chips: speed and memory

Run `.venv/bin/python docs/contributing/benchmarks/2026-11-06-lazy-chips/benchmark.py`.

## Method

- Exact pre-change Chips source is frozen in `eager_baseline.py` for comparison.
- Each sample runs in a fresh subprocess. Three repeats per method/workload/size, alternating method order.
- Local tiled, uncompressed uint16 TIFFs: three bands, 2048² or 4096² pixels. Chips converts to float32.
- 256² chips, no overlap; 512² Dask chunks; four computation threads. Each result is checked for shape, dtype and pixel values.
- Sparse reads select 16 evenly spaced tiles; full reads visit every tile without retaining outputs.
- Timings exclude imports, opening the source, validation of returned pixels and fixture creation. Total includes Chips construction and all requested chip reads.
- Peak RSS is the OS process high-water mark. Added peak subtracts the pre-construction high-water mark. Absolute process memory includes Python/geospatial dependencies.
- GDAL cache defaults are retained, and filesystem caches are not cleared. Full lazy scans can grow backend caches and allocator pools even though Chips retains no pixel arrays.
- This is a local, single-process input benchmark; it does not measure model execution, DataLoader workers/prefetch, remote I/O, overlapping tiles or output merging.

## Median results

| Raster | Read | Method | Setup (s) | Reads (s) | Total (s) | Added peak RSS (MiB) | Absolute peak (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2048² | sparse (16) | eager | 0.190 | 0.003 | 0.193 | 174.2 | 438.8 |
| 2048² | sparse (16) | lazy | 0.012 | 0.247 | 0.259 | 52.6 | 317.3 |
| 2048² | full (64) | eager | 0.243 | 0.010 | 0.253 | 174.2 | 439.0 |
| 2048² | full (64) | lazy | 0.010 | 0.772 | 0.782 | 80.2 | 345.0 |
| 4096² | sparse (16) | eager | 0.592 | 0.004 | 0.594 | 680.2 | 945.4 |
| 4096² | sparse (16) | lazy | 0.005 | 0.210 | 0.214 | 43.1 | 308.2 |
| 4096² | full (256) | eager | 0.588 | 0.036 | 0.627 | 680.1 | 945.8 |
| 4096² | full (256) | lazy | 0.005 | 3.112 | 3.117 | 297.4 | 562.9 |

## Conclusions

Construction no longer materializes all input pixels. Sparse access substantially reduces peak memory. On the larger fixture it also reduces latency; on the smaller fixture per-chip overhead makes it slower. A complete scan trades throughput for lower retained input memory: per-chip xarray conversion, Dask scheduling and repeated chunk reads cost more than slicing an already materialized NumPy array.

Do not describe the change as universally faster or OOM-proof. Source/backend caches, chunk sizes, upstream transforms and worker concurrency still matter. Callers with sufficient memory can explicitly compute/persist prepared sources; Chips itself does not choose a full-source cache.

`merge()` remains a full-output-memory operation and is excluded from this benchmark. The 48 MiB and 192 MiB full float32 input arrays illustrate why constructor-wide materialization was unsafe when extended to many or larger sources.

Raw measurements: [results.json](results.json). No performance threshold is enforced in the unit tests.
