# Spatial samples and temporal frames

```python
samples = items.gs.sample()
frames = samples.gs.frame(
    windows={
        "image": {
            "slots": ["-3W", "-2W", "-1W"],
            "reduce": "first",
            "fill": "interpolate",
        },
    },
    reference="label",
    stride=1,
)
chips = frames.gs.chip(128, resolution=10)
image = chips.iloc[0].gs.to_xarray()["image"].dataset
```

`sample()` groups equal spatial footprints, retaining every acquisition and
native grid. It does not select a reference collection or align pixels.
Containing or overlapping footprints are separate groups.

## Slots define model time steps

Each collection's `slots` list describes ordered, non-overlapping bins. The
integer gives the relative position; the unit gives the bin width. At a
January 15 reference, `-2W` covers January 1–8 and `-1W` January 8–15.
Starts are included and ends excluded. `0D` covers the day starting at the
reference instant. An acquisition exactly at the reference is excluded from
`-1D` and included in `0D`; the clock is not rounded to midnight.

Supported units are `s`, `min`, `h`, `D`, `W`, `M` and `Y`. Weeks are seven days;
months and years are relative calendar offsets. `-1W` is a week-long bin,
whereas `-7D` is a one-day bin seven days before the reference.

A reference collection supplies candidate timestamps and the matching
observation directly; it needs no `windows` entry. With `reference=None`,
timestamps are the sorted unique union of the requested collections, newest
first. `stride` counts these candidates before checking completeness. Its
default is the largest slot count, which does not guarantee non-overlapping
calendar windows. Timeless collections cannot supply the clock.

## Multiple acquisitions, gaps and composites

Each populated bin produces one raster time step. Multiple acquisitions
require an explicit `reduce` method. `first` and `last` select whole scenes
in acquisition-time order, preserving their nodata pixels. Other reducers
use the existing temporal operations, including their categorical-data checks.

`fill="interpolate"` linearly fills absent interior bins using only populated
bins in the same collection's requested window. It does not extrapolate,
fill clouded pixels in existing bins, or read observations outside the window.
Two slots cannot interpolate a missing endpoint. Remaining structurally
incomplete frames are omitted with a count warning. Pixel-level missing
values can remain; framing does not eagerly inspect pixels.

Composite start/end coverage must fit wholly inside a bin. A matching weekly
composite works directly; a monthly composite crossing a weekly boundary does
not become weekly data. No implicit temporal clipping or nearest-time matching
is performed. Persist composite coverage when writing TIFFs and rebuilding
catalogs from their paths.

## Lazy reads and persistence

Frames remain GeoDataFrame rows. `reference_time` and the row's temporal bounds
identify the prediction/reference instant. A JSON `frame` column records bin
bounds, selected Item positions and operations. It survives GeoParquet and
chipping. Reads execute these operations lazily using the existing temporal
implementation, returning one Dataset per collection in a DataTree.

Output `time` labels are bin starts, with explicit inclusive
`start_datetime`/`end_datetime` coordinates. Original acquisitions and coverage
remain in the row's Items. Reference observations retain their own metadata.
Chipping declares spatial alignment: supply chip size and resolution to plan a
local grid in each row's shared source CRS, or supply an explicit target GeoBox.
Chip reads honour the recorded grid and resampling before temporal assembly.
Unchipped reads keep native grids by default. Use
`row.gs.to_xarray(resolution=10)` to align a complete frame on the same local
grid used by resolution-based chipping, without building a chip index.
The spatial grid is independent of the temporal clock.
