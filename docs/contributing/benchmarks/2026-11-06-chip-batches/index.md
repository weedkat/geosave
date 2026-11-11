# Chips batch-local reads

## Caller contract

```python
chips = Chips(
    samples, size=256, overlap=64,
    max_block_bytes=32 * 1024**2,
)
loader = DataLoader(chips, batch_size=16, shuffle=False)
```

DataLoader's map-style batch fetch uses `Chips.__getitems__`. Flat integer
indices retain order and duplicates. Nearby windows from one source share a
computed input block; Tiler provides both bounding boxes and chip padding.

The 32 MiB default limits each materialized input block, including variables,
leading dimensions and selected dtype. It does not bound output batch arrays,
conversion temporaries, Dask/backend caches, collation, worker prefetch or Merger
accumulators. A clipped single chip larger than the budget fails before reading.
Unsized cast dtypes are rejected for batching because their itemsize is unknown.

Ordinary indexing and keyed iteration retain individual-window behavior. There
is no persistent cache, direct Rasterio path or automatic change to source grids.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/geosave-mpl .venv/bin/python \
  docs/contributing/benchmarks/2026-11-06-chip-batches/benchmark.py
```

- Three fresh-process repeats per method and access order, alternating method
  order. Timings exclude imports, fixture writes and correctness assertions.
- Local SSD temporary files, warm/uncleared filesystem caches. A three-band
  DEFLATE TIFF has 4096 × 4109 pixels, uint16 source and float32 chip output.
- 256² chips with 64-pixel overlap; 462 chips per scan; batches of 16.
- Same 512² Dask chunks, four threads, 32 MiB GDAL cache and 32 MiB input-block
  budget. Sequential and fixed-seed shuffled full scans are compared.
- Individual baseline fetches a list of 16 chips with single indexing.
  Batched fetch calls `__getitems__` with exactly those same indices.
  Both retain the same output batch, then discard it after verification.
- Every returned chip is checked against a coordinate-derived pixel oracle,
  including nodata and padded edges. Digests must agree between methods.
- Timing covers construction and batch fetching, not Torch collation, model
  execution or DataLoader multiprocessing. Unit tests separately exercise
  actual default DataLoader batch dispatch.
- Peak RSS includes imports, caches and verification temporaries; added peak
  subtracts the pre-construction process high-water mark. This is not just the
  input-block allocation.
- The pixel oracle and memory sampler are reused from the earlier four-reader
  study. Raw per-run data is stored in `results.json`.

## Results

Median of three runs. Times include construction plus chip fetches.

| Order | Method | Time s | Added peak RSS MiB | Absolute peak RSS MiB |
| --- | --- | ---: | ---: | ---: |
| sequential | individual | 11.907 | 79.9 | 327.1 |
| sequential | batch | 4.928 | 119.9 | 367.1 |
| shuffled | individual | 21.108 | 79.4 | 326.7 |
| shuffled | batch | 19.987 | 82.0 | 329.3 |

Sequential batches were 2.42× faster, at roughly 40 MiB additional peak RSS.
Shuffled batches improved only about 5%; this optimization primarily benefits
spatially nearby requests. These results do not establish training throughput
with worker prefetch, Torch collation or a model running concurrently.

Focused Chips and model-processing tests: **63 passed, 17 warnings**. Coverage
includes DataLoader dispatch, shared-chunk computation, request order,
independent duplicates, boundary padding and byte-budget preflight.
