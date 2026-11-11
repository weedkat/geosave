"""Render benchmark medians and estimated cache break-even counts."""

import json
import math
from pathlib import Path
import statistics

folder = Path(__file__).parent
report = json.loads((folder / "results.json").read_text())
groups = {}
builds = []
for row in report["results"]:
    if row["method"] == "cache_create":
        builds.append(row)
    else:
        groups.setdefault((row["workload"], row["overlap"], row["method"]), []).append(
            row
        )


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def delta(rows, key):
    return statistics.median(row["after"][key] - row["before"][key] for row in rows)


lines = [
    "## Results",
    "",
    "Medians of three fresh-process runs. Added peak is relative to the process",
    "high-water mark before opening/planning; it is not total process memory.",
    "Anon/file columns show changes in resident memory at the end of reading,",
    "not separate peak values.",
    "",
    "| Read | Overlap | Method | Setup s | Read s | Total s | Added peak MiB | Added anon MiB | Added file MiB |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for workload in ("sparse", "full"):
    for overlap in (0, 64):
        for method in ("chips", "blocks", "rasterio", "npy"):
            rows = groups[workload, overlap, method]
            lines.append(
                f"| {workload} ({rows[0]['chips']}) | {overlap} | {method}"
                f" | {median(rows, 'setup_s'):.3f} | {median(rows, 'read_s'):.3f}"
                f" | {median(rows, 'total_s'):.3f} | {delta(rows, 'peak_mib'):.1f}"
                f" | {delta(rows, 'RssAnon_mib'):.1f} | {delta(rows, 'RssFile_mib'):.1f} |"
            )

build = median(builds, "seconds")
lines += [
    "",
    "## Cache cost and estimated reuse break-even",
    "",
    f"- Cache creation: **{build:.3f} s** median, including array flush/fsync.",
    f"- Source TIFF: **{report['source_bytes'] / 2**20:.1f} MiB**.",
    f"- Additional float32 NPY: **{report['cache_bytes'] / 2**20:.1f} MiB**, plus a small JSON sidecar.",
    f"- Cache-build absolute peak RSS: **{statistics.median(row['after']['peak_mib'] for row in builds):.1f} MiB**.",
    "",
    "Break-even counts estimate the first N satisfying",
    "cache_build + N × npy_read < N × alternative_read, using measured medians.",
    "They assume unchanged input/preprocessing and similar warm-cache conditions;",
    "they are estimates, not measured multi-epoch training runs.",
    "",
    "| Read | Overlap | NPY first pass incl. creation s | Versus current Chips | Versus blocks | Versus Rasterio |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for workload in ("sparse", "full"):
    for overlap in (0, 64):
        cached = median(groups[workload, overlap, "npy"], "total_s")
        counts = []
        for method in ("chips", "blocks", "rasterio"):
            difference = median(groups[workload, overlap, method], "total_s") - cached
            counts.append(
                str(math.floor(build / difference) + 1)
                if difference > 0
                else "no benefit"
            )
        lines.append(
            f"| {workload} | {overlap} | {build + cached:.3f} | {' | '.join(counts)} |"
        )

lines += [
    "",
    "## Interpretation",
    "",
    "- Bounded blocks improve full-scene scanning without replacing xarray/Dask. Sparse reads can compute extra pixels inside group bounding rectangles, so the benefit depends on locality.",
    "- Direct Rasterio is a strong route for already-prepared TIFFs. It omits per-chip xarray conversion and Dask graph execution, but does not implement arbitrary lazy preprocessing, multi-source alignment or temporal assembly.",
    "- Memory-mapped NPY is fast when a valid cache already exists. It does not mean zero memory: page-backed residency can exceed direct Rasterio's RSS, especially for a full scan. Most of that increase is file-backed and reclaimable.",
    "- Creating a full-scene NPY solely for one pass is a different decision from reusing a prepared cache. The table above includes that up-front cost; storage and invalidation also matter.",
    "- Keep generic lazy Chips indexing. The next general optimization to consider is an explicit bounded sequential/batch read path. Use a direct file reader only where the caller's input contract is an already-prepared raster.",
    "- This is not evidence that Dask is universally slow. Timings cover this implementation, format, chunk layout and local workload; no profiler isolates a single root cause.",
    "",
    "## Verification and limits",
    "",
    "- Small smoke: 16 read cases plus one complete cache build/validation passed.",
    "- Main study: 48 read cases plus three complete cache builds passed. All chip values, shapes, dtype and padding matched the independent oracle; output digests agreed across approaches.",
    "- Ruff and formatting checks passed. No production library code changed.",
    "- The first large attempt failed with SIGBUS while writing a cache into nearly-full /tmp tmpfs. The completed measurements use workspace SSD temporary files, cleaned up automatically. No unrelated files were removed.",
    "- This does not benchmark saved-chip collections, cache invalidation, remote storage, cold caches, GPU utilization, multiple DataLoader workers or full-output merger memory.",
]
path = folder / "index.md"
prefix = path.read_text().split("\n## Results")[0]
path.write_text(prefix.rstrip() + "\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
