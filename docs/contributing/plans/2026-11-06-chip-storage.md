# Chips disk preparation implementation plan

> Historical: chip caching, saving and the separate disk-backed reader were
> removed on 2026-11-07. Examples and benchmark scripts below describe the
> removed implementation. Storage is deferred until model-context ownership
> is settled; see the [current Chips contract](../specs/2026-11-06-chips.md).

Goal: compare lazy Chips, temporary prepared-source caches and persistent final-chip stores.

The user approved the cache/save design and requested implementation and performance tests.
Use `Chips.cache(directory=None, workers=4)` as a context manager yielding a disk-backed
chip reader. `Chips.save(path, workers=4)` writes final tiles and returns the local path;
`Chips.open(path)` reopens either store. Storage implementation belongs in `io/chips.py`.
No source arrays or graphs are serialized. The reader owns mmap lifetime, process-local
reopening, Tiler layout and copied NumPy results. Metadata records source index, original
shape, ordered axes/variables, nonspatial coordinates, nodata and affine/CRS when present.
Temporary cleanup happens on context exit; callers must stop DataLoader workers first.
Persistent output refuses existing destinations and cleans incomplete writes on failure.
Only local disk locations are supported. Existing lazy and merge behavior is preserved.

## Tasks

- [x] Add failing round-trip tests for cached and saved chips: multisource lazy inputs,
  padding, order, copies, geobox, dtype, budget, cleanup/failure and spawned DataLoader.
- [x] Implement bounded source block writes and bounded tile batches with one Dask pool.
  Reopen read-only maps per process, excluding maps from pickle state.
- [x] Smoke-test all storage paths before settling the design. Run focused Chips,
  storage and model tests, lint and targeted type checks.
- [x] Benchmark fresh processes: no cache, cache and persistent chips; preparation,
  shuffled repeated epochs, disk bytes and process memory. Compare 1/2/4/8 preparation
  threads on the same lazy numeric pipeline; report limits and complete results.
- [x] Update API documentation, spec and review; independently review changed code.

## Review focus

- Source geometry and leading dimension identity survive source-free reopen.
- Multiple workers never pickle an array's pixels or retain source graphs.
- Empty/drop inventories and huge tiles do not bypass preparation bounds.
- Invalid inputs and upstream computation failures leave no publishable artifact.
- Temporary context lifetime and independent returned arrays prevent dangling maps.

Verification is scoped to affected workflows; unrelated dirty work remains untouched.

## Execution evidence

- Initial seven storage tests failed because cache/save/open were absent.
- Storage smoke: seven tests passed; spawned-worker test passed separately after
  sandbox Unix socket denial was resolved with an unrestricted test run.
- Small benchmark: all three paths produced identical pixels and checksums.
- Independent static review found one dtype issue: structured/subarray dtypes
  cannot be represented by dtype.str. A pre-computation rejection regression
  is being added. No other concrete findings or deferred minor issues.

- Compound dtype regression: two failures before the fix; full focused run afterward: 73 passed, 21 warnings. Targeted typing and lint passed.

- Capacity preflight regression failed before implementation; final full focused run: 74 passed, 21 warnings. The system tmpfs was full, so all real storage measurements used workspace SSD. The free-space check is not a reservation against concurrent writers.
- Performance complete: nine four-thread comparison runs plus eighteen 1/2/8-thread runs, all with pixel equality and matching digests within each study. Final small smoke reruns after capacity hardening.
