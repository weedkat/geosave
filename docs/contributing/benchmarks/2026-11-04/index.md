# Local performance measurements — 2026-11-04

GeoSave's metadata-only conversions are inexpensive and remain lazy in these
cases. Full multiband COG reading and excessive small Dask chunks are the
clearest measured optimization opportunities. NetCDF adds little overhead over
xarray here. Zarr could not be timed: both GeoSave and native xarray workers
failed to finish their first write within a 40-second process deadline.

These are measurements of the current working tree on one development machine,
not release-wide performance guarantees or comparisons with a previous version.
No production implementation or template was changed.

## Reproduce and inspect

Run from the repository root, **sequentially**:

```bash
MPLCONFIGDIR=/tmp/geosave-perf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python docs/contributing/benchmarks/2026-11-04/benchmark.py \
  --output docs/contributing/benchmarks/2026-11-04/results.json
MPLCONFIGDIR=/tmp/geosave-perf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python docs/contributing/benchmarks/2026-11-04/cog_readers.py
MPLCONFIGDIR=/tmp/geosave-perf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python docs/contributing/benchmarks/2026-11-04/extended.py
MPLCONFIGDIR=/tmp/geosave-perf-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --no-sync python docs/contributing/benchmarks/2026-11-04/cross_crs.py
```

The four scripts are benchmark artifacts, not a new library API or CI gate.
They use temporary files and write raw results beside this report. Rerunning
overwrites the result JSON files; copy them first to retain this snapshot.

- [Main runner](benchmark.py) and [raw results](results.json): individual repeats,
  min/median/max, Dask task probes, versions, import timings, format results.
- [COG reader comparison](cog_readers.py) and [raw results](cog-results.json).
- [Cross-CRS runner](cross_crs.py) and [timings/reference comparisons](cross-crs-results.json).
- [Chunk/crop/catalog runner](extended.py) and [raw results](extended-results.json).

## Method and limits

- AMD Ryzen 5 4500U, six logical CPUs; Linux; Python 3.14.9. Full dependency
  versions and the source fingerprint are in `results.json`.
- GeoSave 0.3.0 working-tree code; xarray 2026.4.0, Dask 2026.3.0, Rasterio 1.5.0,
  rioxarray 0.22.0, odc-geo 0.5.1, torch 2.11.0, Zarr 3.2.1, netCDF4 1.7.4.
- Deterministic four-band `uint16` values from 1–9999; EPSG:32633, 10 m grid;
  512×512 and 2048×2048 scenes: **2 and 32 MiB of source pixels**. Random values
  avoid unrealistically easy compression of all-zero fixtures. Nodata is zero.
- Dask uses four threads; torch, OpenBLAS and OpenMP are limited to one thread.
  Default in-memory chunks are 256×256. Codec/backend threads are not otherwise
  forced, so this is not a universal single-thread comparison.
- One untimed warmup, then five timed repeats; catalogs and store workers use
  three repeats. Fixture creation, correctness checks and cleanup after each
  timing are excluded unless explicitly part of the operation. Fresh open/load
  helpers include opening and closing their handles.
- Values, shapes, grids, dtype where relevant, or catalog/sample counts are
  checked after timing. A separate Dask Callback invocation counts executed
  tasks, avoiding callback instrumentation inside the reported times.
- **Warm filesystem cache:** no privileged cache dropping, no durable `fsync`
  guarantee, and no remote latency. Output goes to local temporary storage.
  COG compression/overviews use current writer defaults. These are repeated
  software-path measurements, not sustained disk bandwidth.
- Inputs to in-memory transforms are already NumPy-backed or Dask-wrapped
  NumPy data; they measure computation/scheduling, not acquisition from disk.
  In-memory crop results do not establish cloud bytes transferred.
- `raw_mib_per_s` in JSON normalizes against the full input scene. For subset,
  single-feature or multiresolution operations it is **not actual memory or I/O
  bandwidth**. Use the timings, task counts and stated shapes for those cases.
- RSS fields are lifetime high-water marks, including imports, fixtures and
  allocator history; fork/exec workers can inherit a prior watermark. They
  cannot establish per-operation memory cost or leaks. No reliable allocation
  profile, cold-disk, GPU, distributed, or cloud benchmark is claimed.

Fresh-process import medians (three processes each; interpreter launch excluded):
`geosave` **1.273 s**, `torch` **1.077 s**. Tensor operation timings below already
have torch imported. Do not promise their warm timings for a first tensor call.

## Raster operations

All values are median **milliseconds**. `plan` constructs deferred work;
`compute` executes an already-built graph. End-to-end latency needs both when
both stages are used.

| Operation | 512×512, 2 MiB | 2048×2048, 32 MiB |
| --- | ---: | ---: |
| `to_datatree` | 0.76 | 0.74 |
| `to_dataset.same_grid` | 3.50 | 3.48 |
| `to_dataset.multires.plan` | 11.13 | 15.57 |
| `to_dataset.multires.compute` | 18.24 | 147.02 |
| `decode.plan` | 9.45 | 9.16 |
| `decode.compute` | 10.05 | 137.90 |
| `ndvi.plan` | 12.05 | 11.63 |
| `ndvi.compute` | 8.65 | 126.59 |
| `reproject.plan` | 11.00 | 13.34 |
| `reproject.compute` | 12.64 | 149.52 |
| `to_numpy.eager` | 1.09 | 10.73 |
| `to_tensor.eager.uint16` | 1.12 | 11.15 |
| `to_tensor.eager.float32` | 1.84 | 58.41 |
| `to_tensor.lazy.uint16` | 7.03 | 65.04 |
| `to_tensor.tree` | 6.91 | 62.23 |
| `chips.layout` | 0.50 | 1.22 |
| `chips.select` | 12.85 | 13.03 |
| `chips.to_tensor` | 18.59 | 10.52 |
| `cog.write` | 46.43 | 947.61 |
| `cog.open` | 20.39 | 21.61 |
| `cog.open_load` | 101.10 | 959.11 |
| `cog.open_tensor` | 94.50 | 954.90 |
| `cog.window_tensor` | 96.61 | 99.20 |
| `baseline.rasterio.read` | 8.67 | 132.20 |

Multiresolution conversion adds a fifth variable on a half-resolution grid and
upsamples it onto the finest grid. Reprojection here is nearest-neighbour
half-resolution resampling on the same CRS, **not** a cross-CRS reprojection
benchmark. Chip/window conversion always returns a 256×256 four-band tensor,
regardless of parent scene size. COG open/load/tensor returns all four bands.

All tested tree conversions, decoding/NDVI/warp plans, chip layout and chip
selection executed **zero Dask tasks**. Larger pixels chiefly affect materialized
work, not simple tree wrapping. Same-grid `to_dataset()` costs about 3.5 ms at
both sizes; that does not justify replacing native DataTrees with wrappers.

The 2048 COG window uses nine Dask tasks, versus 129 for the full tensor. Its
~99 ms latency shows that selected windows avoid full-scene execution in this
local case; opening, band readers, decoding and fixed scheduling costs remain.

### Tensor conversion and chunk sensitivity

At 2048×2048, the first run's eager uint16 tensor takes 11.15 ms versus 5.32 ms
for direct NumPy stacking plus `torch.from_numpy`, and 5.79 ms for native xarray
stacking. The simple baselines do not perform GeoSave's named-axis/dtype checks.

Float32 conversion is sensitive to process history: the main run measured
58.41 ms, while the separate focused run measured **17.52 ms** for GeoSave and
**11.25 ms** for direct `np.stack(..., dtype="float32")` plus torch conversion.
The broad number should not be used as a stable regression threshold. The
source currently stacks before casting; one-allocation conversion is a candidate
to profile while preserving channel order, axes, contiguity and writable values.

| In-memory chunk | Full tensor ms | Full tensor Dask tasks | Crop plan ms | 256×256 cropped tensor ms |
| --- | ---: | ---: | ---: | ---: |
| 128×128 | 205.21 | 2049 | 10.45 | 11.54 |
| 256×256 | 59.08 | 513 | 8.94 | 4.94 |
| 512×512 | 24.34 | 129 | 8.95 | 4.73 |
| 1024×1024 | 19.35 | 33 | 8.81 | 5.05 |

All crop plans execute zero tasks. Full-scene conversion falls from 205 ms with
2,049 tasks to 19 ms with 33 tasks. Cropping first gives approximately 5 ms at
256–1024 chunks for this in-memory fixture. This supports documenting and testing
chunk choice for the actual workload; it does **not** justify globally increasing
chunks, since remote chip reads and concurrent workers have different needs.

## Isolating the COG reading cost

Same 32 MiB input, same multiband COG, same output values. The file uses 512×512
blocks, LZW compression and pixel interleaving. Inspect exact metadata in
`cog-results.json`. The random-data COG is 52.48 MiB on disk for 32 MiB of
source pixels, including its encoding and overviews; compression is not assumed
to shrink this fixture. Results are from the separate controlled reader run; they
need not exactly equal the broad run above.

| Reader configuration | Median ms | Min–max ms | Dask tasks |
| --- | ---: | ---: | ---: |
| `geosave.chunks=None` | 473.70 | 464.89–563.14 | 0 |
| `geosave.chunks={}` | 876.84 | 663.79–947.02 | 129 |
| `geosave.chunks={'y': 1024, 'x': 1024}` | 668.14 | 586.73–703.23 | 33 |
| `geosave.chunks={'y': 2048, 'x': 2048}` | 556.06 | 553.15–587.86 | 9 |
| `rioxarray.split=False.chunks=None` | 129.41 | 127.73–145.59 | 0 |
| `rioxarray.split=False.chunks={}` | 167.72 | 161.58–185.27 | 2 |
| `rioxarray.split=True.chunks=None` | 501.42 | 499.78–511.21 | 0 |
| `rioxarray.split=True.chunks={}` | 549.25 | 527.41–575.34 | 9 |
| `geosave.separate_band_files` | 339.84 | 275.75–382.13 | 132 |

The strongest controlled distinction is **129 ms** for one non-Dask rioxarray
multiband array versus **501 ms** for separate band variables without Dask.
GeoSave's non-Dask path is comparable at **474 ms**. Its default Dask path is
**877 ms**, with finer spatial chunking and 129 tasks; rioxarray's `{}` defaults
here produce different graphs, so that comparison is not wrapper overhead alone.

`io/rasterio.py` opens bands as variables and sets spatial chunks from file
blocks. Those choices implicate read layout, repeated band access and scheduling.
The measurements do not count codec invocations, so repeated decompression is a
hypothesis to profile, not a proven count. Separate single-band COGs take 340 ms
in this sequential full-read case, but have different compression and access
tradeoffs; do not change publication defaults from this one fixture.

**Recommended experiment:** internally open one multiband array and expose its
bands as Dataset variables while sharing the underlying reads. Verify band names,
per-band attrs, nodata/packing, band-selective reads, close ownership, overview
selection and lazy windows. Benchmark pixel- and band-interleaved files with the
same requested chunk shapes. Retain native Dataset values at the public interface.

## Cross-CRS performance and numerical comparison

A separate 1024×1024 four-band uint16 source (8 MiB), chunked 256×256, is
reprojected from EPSG:32633 to EPSG:3857 at 10 m. The destination is 1461×1457.
Three timed repeats follow a warmup. Planning runs zero Dask tasks; each
computation executes 148. Comparison uses the red band and an independently
allocated whole-array Rasterio destination with the same grids/kernel/nodata.

| Kernel | Plan ms | Compute ms | Differing valid pixels vs whole-array reference | Maximum absolute difference | Nodata-mask disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| nearest | 183.09 | 522.90 | 64,992 / 2,113,983 | 9940 | 178 |
| bilinear | 183.96 | 433.09 | 2,088,328 / 2,113,983 | 474 | 178 |

The eager GeoSave path has **zero differing values** against Rasterio on the
compared valid pixels for both kernels. The lazy path differs from that eager
output. Nearest differs at about 3.1% of valid comparison pixels; bilinear at
about 98.8%. Integer-valued, high-frequency random inputs are sensitive to
sample-location/interpolation differences, so these rates are not a measure
of real-world semantic error. Absolute differences are stored pixel units,
not metres. Nonetheless, timing and correct output shape alone are insufficient.

This extends the earlier known warp limitation to a reproduced nearest-neighbour
case. Reduce this fixture against the ODC lazy warp implementation, examine
chunk-local transform/approximation and neighbourhood choices, and decide
numerical tolerances from the intended operation. Do not automatically force
whole-scene eager computation or loosen tests simply to obtain a pass.

## NetCDF and Zarr

512×512 four-band scene; three repeats. NetCDF is uncompressed default storage,
so its numbers are not an encoding-equivalent speed ranking against compressed
COG. Native baseline uses xarray with the same NetCDF engine.

| Operation | GeoSave ms | Native xarray ms |
| --- | ---: | ---: |
| NetCDF write | 6.25 | 5.85 |
| NetCDF open | 6.48 | 5.16 |
| NetCDF open_load | 12.54 | 11.82 |

NetCDF round-trip values/grid checks pass. GeoSave's wrapper overhead is small
at this size. Both Zarr workers hit their **40 s process deadline during the
first write**, before `WRITE_COMPLETE`; no successful write/open throughput is
reported. Deadlines include child startup/imports. Because native xarray also
stalls, first isolate the installed Zarr/backend/runtime interaction rather than
redesign GeoSave persistence based on this environment.

## Catalog metadata workloads

One collection, 50 dates per Region, disjoint geographic footprints; 2, 20 and
100 Regions. `sample(pair=["optical"], window=3)` yields 96, 960 and 4,800
Windows. Assets use nonresolving example URLs and **are never opened**. This
measures metadata conversion, grouping and GeoParquet persistence, not STAC
service latency or raster loading. Catalogs have a small fixed metadata schema.

| Items | `from_items` ms | `sample` ms | GeoParquet write ms | Open ms |
| --- | ---: | ---: | ---: | ---: |
| 100 | 33.05 | 36.77 | 62.99 | 26.16 |
| 1,000 | 92.88 | 294.98 | 360.74 | 45.96 |
| 5,000 | 370.46 | 1,572.59 | 1,921.45 | 133.55 |

Sampling takes 1.57 s and writing 1.92 s at 5,000 Items. The roughly proportional
scaling in this fixture does not establish pathological complexity or justify
moving the sampling algorithm to a different module. Overlapping swaths, many
collections, rich metadata and repeated pairing may have different costs.
Profile Item/Arrow serialization and Region/Pair planning on an actual large
catalog before adding caching or parallelism.

## What to tackle first

1. **Lazy cross-CRS correctness:** resolve or explicitly bound the observed
   eager/lazy discrepancies before tuning warp throughput.
2. **Shared multiband COG reads:** largest measured gap against a native path
   producing the same values; preserve per-band semantics and resource ownership.
3. **Workload-aware chunks and crop-before-tensor guidance:** demonstrated task
   overhead, with no API redesign needed. Benchmark actual on-disk chunks and
   worker concurrency before changing defaults.
4. **Tensor stacking/casting allocations:** a smaller, measurable opportunity;
   use the focused comparison and profile allocations before changing code.
5. **Zarr runtime diagnosis:** restore a complete native and GeoSave round trip
   in this environment, then measure it. A timeout is not a throughput result.
6. **Catalog profiling only where scale requires it:** preserve current domain
   ownership; this fixture does not support a broad sampling rewrite.

The correctness issues in the [project review](../../review-2026-11-04.md)
still take priority: a fast omitted raster, misaligned sample, or incomplete
prediction is not an improvement. Other CRS/kernel combinations, Zarr completion, cold-disk
I/O, remote byte ranges, browser rendering, GPU transfers, large time axes,
and peak allocation attribution remain unmeasured here.
