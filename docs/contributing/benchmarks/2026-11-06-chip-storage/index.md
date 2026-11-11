# Temporary caches and persistent chips

> Historical: chip caching, saving and the separate disk-backed reader were
> removed on 2026-11-07. Examples and benchmark scripts below describe the
> removed implementation. Storage is deferred until model-context ownership
> is settled; see the [current Chips contract](../../specs/2026-11-06-chips.md).

```python
chips = Chips(samples, size=256, overlap=64)

# Prepare once for this run. Stop workers before leaving the context.
with chips.cache(directory="./scratch", workers=4) as cached:
    loader = DataLoader(cached, batch_size=16, shuffle=True)
    trainer.fit(model, loader)

# Save final padded chips for later runs; destination must not exist.
chips.save("./training-chips", workers=4)
with Chips.open("./training-chips") as saved:
    loader = DataLoader(saved, batch_size=16, shuffle=True)
    trainer.fit(model, loader)
```

Temporary caches store each prepared source once. Persistent stores contain
final tiles and repeat overlapping pixels and padding. Both readers return
writable copies, retain positional source/tile IDs and affine grids, and reopen
read-only memory maps per worker process. Neither retains the input xarray graph.
The original Chips object still owns its lazy sources and prediction merging.

The local directory includes a versioned JSON manifest and one NPY array per
source. Metadata records source shape, ordered axes/variables, nonspatial
coordinate values, nodata and optional CRS/affine. It is a chip training artifact,
not a full xarray serialization: arbitrary attributes and spatial coordinates
are not retained. Persistent destinations are never overwritten. Failed writes
remove the newly created incomplete directory. A manifest is published only
when all arrays have been flushed and fsynced. Process termination can leave an
incomplete directory, which must be removed manually before retrying.

`max_block_bytes` limits each materialized input block. For persistent export,
it also limits the output tile batch separately; these can coexist. Upstream
chunks/intermediates, allocator caches, tensor collation and resident mapped
pages are outside that limit. A single padded tile exceeding it is rejected.
Available disk capacity is checked before computing pixels; this is a snapshot, not a reservation against other writers. All writes use a single Dask thread pool controlled by `workers`; there is no
nested pool or computation of the entire scene into RAM.

Choose a scratch directory on disk. The system temporary directory can be a
small RAM-backed filesystem; the benchmark uses the workspace SSD. The default
`directory=None` follows the system temporary directory setting. Cached data
is a snapshot: source changes are not tracked, and random augmentation belongs
after reading. Close DataLoader workers before deleting a temporary cache;
`persistent_workers=True` requires caller-managed worker shutdown.

## Reproduce

```bash
.venv/bin/python docs/contributing/benchmarks/2026-11-06-chip-storage/benchmark.py
```

Smoke: `--size 512 --epochs 2 --repeats 1 --output <local-path>/smoke.json`.
Thread sweep: `--thread-counts 1 2 8 --epochs 1 --output <local-path>/threads.json`.

## Method

- Each method/repeat runs in a fresh process; three repeats, alternating order.
- Local SSD, warm/uncleared filesystem caches. Source is a three-band uint16
  DEFLATE TIFF of 4096 × 4109 pixels; Dask chunks are 512², GDAL cache 32 MiB.
- All methods use the same lazy xarray arithmetic and nodata replacement before
  chipping, so cache creation includes real xarray/Dask execution.
- 256² float32 chips, overlap 64, reflect padding, batches of 16; 462 chips/epoch.
- Three differently shuffled epochs per run, using actual PyTorch DataLoader
  with default tensor collation and zero loader subprocesses. Preparation uses
  four Dask threads. Torch/OpenMP numerical threads are held at one.
- Timings separate source setup, cache/export creation (including flush/fsync)
  and each epoch's iterator creation/fetches. Imports, fixture writes and pixel
  verification are excluded. No model, GPU transfer or augmentation is timed.
- Every chip is checked against an independent coordinate formula, including
  nodata replacement and edge padding. Digests must match across all methods.
- Process RSS is recorded before setup, after preparation and after epochs.
  Peaks include verification allocations, imports and file-backed resident pages;
  they are not bounds on live NumPy arrays. Imports include Torch in all methods.
- Actual spawned worker correctness is tested separately. Worker throughput,
  remote input, cold cache behavior and expensive reprojection are not inferred.

## Results

Median of three fresh-process repeats. Epoch time is the median of each run's three differently shuffled epochs, then the median across repeats.

| Mode | Preparation s | Epoch s | Setup + preparation + 3 epochs s | Disk MiB | Added peak RSS MiB | Absolute peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lazy | 0.000 | 19.605 | 61.134 | 0.0 | 101.4 | 811.9 |
| cache | 3.042 | 0.170 | 3.636 | 192.6 | 404.9 | 1114.8 |
| saved | 5.638 | 0.124 | 6.148 | 346.5 | 519.1 | 1228.5 |

For this workload, either preparation pays back within the first full epoch. The temporary cache costs less to prepare and occupies less disk. Saving final chips is slightly faster per epoch and supports reuse across runs, but repeats overlapping pixels. These are data-fetch timings, not end-to-end model training speedups.

Disk preparation does not eliminate memory pressure. Compared with the baseline, mapped pages increase resident memory; Dask/NumPy allocation and backend caches also remain. After epochs, mapped file residency increased by about 211 MiB for source caches and 365 MiB for saved chips, versus about 18 MiB without preparation. File-backed pages are reclaimable, but still count in RSS. No process-wide memory bound is promised.

The earlier 0.94 s cache-build measurement used direct Rasterio writes without this xarray pipeline. The 3.04 s result here measures the implemented generic xarray path, including arithmetic, nodata replacement and metadata.

## Verification

- Smoke: three methods, two epochs each, exact pixel equality and matching digests.
- Main: nine isolated runs, 27 shuffled epochs; every pixel checked and all digests matched.
- Focused storage, Chips and model-processing tests: **74 passed, 21 warnings**.
- Spawned DataLoader test uses two workers for both storage layouts. Sandbox tensor-transfer sockets were denied; the full focused run passed with those sockets enabled.
- Targeted Ruff, formatting, type checks and whitespace checks passed. The complete library suite was not run.
- Independent review found compound dtype metadata loss; structured/subarray dtypes now fail preflight, with a regression verified failing before the fix and passing afterward.


## Preparation threads

Median preparation seconds across three repeats on the same 4096 × 4109 fixture. Four-thread values come from the main run; 1/2/8-thread runs each verify one shuffled epoch after preparation.

| Threads | Temporary source cache s | Persistent final chips s |
| ---: | ---: | ---: |
| 1 | 2.786 | 4.925 |
| 2 | 2.670 | 5.166 |
| 4 | 3.042 | 5.638 |
| 8 | 3.276 | 6.755 |

More threads did not improve this workload consistently. One or two threads were fastest; eight increased preparation time. The API keeps concurrency explicit rather than assuming that more workers help. A heavier lazy pipeline or remote backend may behave differently and needs its own measurement.

The thread sweep completed 18 isolated runs with identical pixels and matching one-epoch digests. The four-thread main run completed nine runs and 27 epochs. Raw results are in `results.json` and `threads.json`.

Large timings were recorded before the final capacity preflight was added; that check does not change the compute/write path. The final small smoke reruns all three paths after hardening.

These measurements predate persistent mapping release. Persistent export now
closes completed batches, and saved-chip reads close maps after copying.
See the [production memory follow-up](../2026-11-07-chip-memory/index.md).
Temporary source-cache behavior is unchanged.
