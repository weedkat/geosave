# Chip reader feasibility: per-chip, bounded blocks, Rasterio and NumPy mmap

This is a prototype comparison, not a change to the library's reader API.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python \
  docs/contributing/benchmarks/2026-11-06-chip-readers/benchmark.py
```

A small smoke run is available with `--size 512 --repeats 1 --output /tmp/chip-smoke.json`.
The standard run writes raw samples and environment details to `results.json`.

## Compared operations

| Method | Implementation | What it requires |
| --- | --- | --- |
| `chips` | Current Chips: one xarray slice and NumPy conversion per chip | Prepared xarray source |
| `blocks` | Group chip origins into 1024² regions, compute each group's bounding window once, extract its NumPy chips | Spatial scheduling; block bounds include chip overhang |
| `rasterio` | One persistent reader, direct window reads into float32, native Tiler padding | Already-prepared raster file |
| `npy` | Memory-map a prepared float32 array; copy chip windows and pad | An existing prepared cache plus separate metadata |

Block grouping retains global tile IDs and uses identical chip positions and
native Tiler boundary padding. The group can extend beyond 1024 pixels to include
its chips' overhang. Only one computed region is retained at a time. All methods
use the same spatially grouped chip order, so order alone does not explain their
differences. This prototype does not promise random-access block caching.

## Method and limits

- Temporary fixtures live beside the output JSON (the workspace SSD in this run),
  avoiding the capacity-limited RAM-backed `/tmp`.
- Local DEFLATE tiled GeoTIFF, three uint16 bands, 4096 rows × 4109 columns.
  Pixel values vary with position and band; explicit nodata is present.
- All output chips are float32; nodata remains the original 65535 sentinel.
  No scale decoding, masking, reprojection or other preprocessing is performed.
- 256² chips with overlap 0 and 64. The nonmultiple raster width exercises
  clipped/padded edge chips.
- Sparse reads select 16 fixed random chip IDs. Full scans visit every chip.
- Dask chunks are 512² with four computation threads. Every method uses the
  same 32 MiB GDAL cache limit. Rasterio reads are serial within each worker.
- Each sample is a fresh process. Three repetitions rotate method order.
  Filesystem caches are not cleared; this is not a cold-disk/network benchmark.
- Timings include opening/planning, reader work, NumPy materialization and
  closing. They exclude imports, fixture creation, checksums and correctness
  assertions. No output chips are retained.
- Every chip is compared pixel-for-pixel against an independent coordinate
  formula with expected padding. Checksums must agree across methods/repetitions.
- Peak RSS is the process high-water mark. Added peak subtracts the high-water
  mark immediately before opening/planning. It includes validation temporaries
  and backend/allocator caches, not just the explicitly retained pixel array.
- Anonymous and file-backed RSS are also recorded after reading. Memory-mapped
  file pages are reclaimable, but they still count toward RSS while resident.
- Cache creation runs separately in a fresh process: Rasterio windows write a
  float32 `.npy` map and CRS/transform/nodata/band-order JSON. Timing includes
  flushing the array and fsync of its file. Complete cache validation is untimed.
- Cache build cost is deliberately reported separately and added back for a
  one-pass comparison. The cached data must be invalidated if source pixels or
  preprocessing change. This script does not implement cache invalidation.
- Memory mapping applies to local files here. Remote object stores, multiple
  workers, model execution, augmentation and prediction merging are unmeasured.
- For sources with lazy transformations, direct Rasterio or raw `.npy` slicing
  does not replace that computation. Their results apply to already-prepared data.

## Results

Medians of three fresh-process runs. Added peak is relative to the process
high-water mark before opening/planning; it is not total process memory.
Anon/file columns show changes in resident memory at the end of reading,
not separate peak values.

| Read | Overlap | Method | Setup s | Read s | Total s | Added peak MiB | Added anon MiB | Added file MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse (16) | 0 | chips | 0.085 | 0.509 | 0.594 | 63.3 | 45.8 | 14.2 |
| sparse (16) | 0 | blocks | 0.059 | 0.489 | 0.548 | 86.4 | 66.3 | 14.4 |
| sparse (16) | 0 | rasterio | 0.016 | 0.023 | 0.039 | 28.2 | 16.1 | 12.2 |
| sparse (16) | 0 | npy | 0.003 | 0.014 | 0.020 | 133.3 | 0.5 | 128.9 |
| sparse (16) | 64 | chips | 0.064 | 0.642 | 0.706 | 65.5 | 47.9 | 14.2 |
| sparse (16) | 64 | blocks | 0.062 | 0.507 | 0.570 | 76.0 | 57.6 | 14.2 |
| sparse (16) | 64 | rasterio | 0.017 | 0.040 | 0.058 | 35.4 | 23.2 | 12.2 |
| sparse (16) | 64 | npy | 0.003 | 0.012 | 0.017 | 121.2 | 0.5 | 116.9 |
| full (272) | 0 | chips | 0.070 | 4.298 | 4.366 | 64.8 | 47.1 | 14.0 |
| full (272) | 0 | blocks | 0.063 | 1.597 | 1.660 | 122.8 | 108.8 | 14.1 |
| full (272) | 0 | rasterio | 0.018 | 0.369 | 0.389 | 54.0 | 42.0 | 11.9 |
| full (272) | 0 | npy | 0.004 | 0.126 | 0.137 | 204.5 | 0.6 | 200.2 |
| full (462) | 64 | chips | 0.066 | 7.604 | 7.670 | 68.2 | 50.9 | 14.0 |
| full (462) | 64 | blocks | 0.064 | 2.719 | 2.786 | 139.6 | 125.7 | 14.1 |
| full (462) | 64 | rasterio | 0.019 | 0.587 | 0.607 | 54.6 | 38.8 | 12.0 |
| full (462) | 64 | npy | 0.005 | 0.213 | 0.225 | 204.5 | 0.6 | 200.2 |

## Cache cost and estimated reuse break-even

- Cache creation: **0.942 s** median, including array flush/fsync.
- Source TIFF: **88.4 MiB**.
- Additional float32 NPY: **192.6 MiB**, plus a small JSON sidecar.
- Cache-build absolute peak RSS: **320.5 MiB**.

Break-even counts estimate the first N satisfying
cache_build + N × npy_read < N × alternative_read, using measured medians.
They assume unchanged input/preprocessing and similar warm-cache conditions;
they are estimates, not measured multi-epoch training runs.

| Read | Overlap | NPY first pass incl. creation s | Versus current Chips | Versus blocks | Versus Rasterio |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse | 0 | 0.962 | 2 | 2 | 49 |
| sparse | 64 | 0.959 | 2 | 2 | 24 |
| full | 0 | 1.079 | 1 | 1 | 4 |
| full | 64 | 1.167 | 1 | 1 | 3 |

## Interpretation

- Bounded blocks improve full-scene scanning without replacing xarray/Dask. Sparse reads can compute extra pixels inside group bounding rectangles, so the benefit depends on locality.
- Direct Rasterio is a strong route for already-prepared TIFFs. It omits per-chip xarray conversion and Dask graph execution, but does not implement arbitrary lazy preprocessing, multi-source alignment or temporal assembly.
- Memory-mapped NPY is fast when a valid cache already exists. It does not mean zero memory: page-backed residency can exceed direct Rasterio's RSS, especially for a full scan. Most of that increase is file-backed and reclaimable.
- Creating a full-scene NPY solely for one pass is a different decision from reusing a prepared cache. The table above includes that up-front cost; storage and invalidation also matter.
- Keep generic lazy Chips indexing. The next general optimization to consider is an explicit bounded sequential/batch read path. Use a direct file reader only where the caller's input contract is an already-prepared raster.
- This is not evidence that Dask is universally slow. Timings cover this implementation, format, chunk layout and local workload; no profiler isolates a single root cause.

## Verification and limits

- Small smoke: 16 read cases plus one complete cache build/validation passed.
- Main study: 48 read cases plus three complete cache builds passed. All chip values, shapes, dtype and padding matched the independent oracle; output digests agreed across approaches.
- Ruff and formatting checks passed. No production library code changed.
- The first large attempt failed with SIGBUS while writing a cache into nearly-full /tmp tmpfs. The completed measurements use workspace SSD temporary files, cleaned up automatically. No unrelated files were removed.
- This does not benchmark saved-chip collections, cache invalidation, remote storage, cold caches, GPU utilization, multiple DataLoader workers or full-output merger memory.
