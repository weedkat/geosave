# Samples from a catalog

```python
items = gs.open_vector("catalog.parquet")
samples = items.gs.sample(
    collections=["s2", "s1", "dem"],
    freq="MS",
    window=3,
    interval="raise",
)
with samples[0].gs.to_datatree() as tree:
    optical = tree["s2"].gs.combine_stac()
```

`sample()` plans membership without opening pixels. Collections independently
populate a common timeline: no collection supplies reference dates for another.
GeoSamples keeps a copy of the catalog table, so later sorting or editing the
source table does not change membership. Nested asset dictionaries are not
recursively copied. `samples.indices` holds source row positions;
`samples[i]` returns those source rows. `samples.slots[i]` records each collection's
membership at each time, including empty lists for missing observations.
`samples.times[i]` gives the ordered slot labels.

## Spatial membership

`match="exact"` groups geometrically equal footprints. The default
`match="overlap"` retains smallest-first footprint discovery: each retained
footprint gathers Items sharing at least `overlap` of its area. A swath may
participate in several groups. `samples.footprints` retains each group's founding
geometry even when temporal selection removes some Items.

With all temporal options at their defaults, `items.gs.sample()` returns
one raw sample per spatial group. Explicit collection selection excludes all
other collections, including static ones. Include a DEM in `collections` when
it should accompany temporal windows. Static Items join once per window and
never create dates or fill gaps in a dated collection. Static membership does not trigger any pixel processing.

## Calendar slots and windows

```text
Observed dates: Jan 12, Mar 19, Apr 03
freq="MS":     January, February (empty), March, April
window=2:      [Jan, Feb]  [Feb, Mar]  [Mar, Apr]
missing="drop":                       [Mar, Apr]
```

`freq` accepts pandas frequency strings or offset objects, with left-closed,
right-open intervals and start labels. The available observations determine
unbounded timeline ends. Use `date_range=(start, end)` to restrict selection;
either bound may be `None`. The lower bound is inclusive and the upper bound
exclusive. Bounds are normalized to UTC. Source Item coverage is never rewritten.

Anchored frequencies take precedence: `freq="MS"` still uses month starts when
`date_range` begins on January 15. The first bucket is labelled January 1, but
observations before January 15 are excluded. Fixed durations use the supplied
lower bound as their anchor, or the Unix epoch when it is absent.

For months beginning on the 15th, use a relative calendar offset:

```python
import pandas as pd

samples = items.gs.sample(
    date_range=("2025-01-15", None),
    freq=pd.DateOffset(months=1),
    window=3,
)
```

Relative `DateOffset` values require an explicit lower bound. Their boundaries
remain January 15, February 15, March 15, and so on, regardless of the first
observation in a spatial group. Explicit bounds retain empty edge periods when
`missing="keep"`; unbounded ends stop at available observations. Without other
temporal options, `date_range` only filters spatial groups by temporal coverage.
Timeless Items remain eligible. For temporal assignment, an Item's nominal
`datetime` still takes precedence over its coverage interval.

`window` counts consecutive slots and `step` advances the window start by that
many slots. There is no origin parameter, reference collection, or signed offset.
Without `freq`, sorted exact representative timestamps define the positions;
interval-only observations require a calendar frequency.

Each selected collection is required. `missing="drop"` omits incomplete windows;
`missing="keep"` retains their empty positions. No valid windows returns empty
`GeoSamples`. Completeness concerns observations, not individual pixel validity.

## Composite intervals

An Item with `datetime` is assigned using that representative instant. Its
start/end coverage is preserved. Interval-only Items are assigned to the slots
their inclusive coverage intersects. A composite crossing slots requires a policy:

| `interval` | Crossing-interval behavior |
|---|---|
| `"raise"` (default) | Identify the Item and ask for an explicit policy. |
| `"first"` | Assign to the earliest intersected slot. |
| `"last"` | Assign to the latest intersected slot. |
| `"overlap"` | Reuse the same observation in every intersected slot. |
| `"drop"` | Exclude the Item from temporal sampling, leaving the catalog unchanged. |

A January 20–February 10 composite crosses January and February monthly slots.
An end timestamp exactly at February 1 also intersects February because Item
endpoints are inclusive. Overlap does not partition pixels or weight their
contribution by days. Incomplete or inverted bounds always raise.

## Loading and preparing pixels

`samples[i]` returns the original catalog rows selected by positional indices.
Use their normal accessor to load observations:

```python
with samples[0].gs.to_datatree() as tree:
    optical = tree["s2"].gs.combine_stac()
    monthly = optical.gs.mask_and_scale().gs.composite("MS", "median")
    # Consume lazy pixels before closing their source tree.
    result = monthly.compute()
```

Loading retains original observation timestamps and native grids. It does not
reduce observations, fill missing slots, or rewrite timestamps to grouping labels.
A collection absent from the selected rows is absent from the loaded tree, even
when its missing membership is recorded in `samples.slots[i]`.

Temporal transforms are explicit operations on native xarray values. They use
raster time coordinates, not the grouper's assignment metadata. In particular,
`interval="last"` changes which window includes an interval-only Item; it does
not change that Item's original timestamp when loaded. Inspect `samples.slots[i]`
if preparation needs to follow those assignments rather than source timestamps.

```python
samples = items.gs.sample(
    collections=["temperature"], freq="D", window=30, missing="keep",
)
with samples[0].gs.to_datatree() as tree:
    observations = tree["temperature"].gs.combine_stac()
    daily = observations.gs.mask_and_scale().gs.composite("D", "mean")
    filled = daily.chunk(time=-1).interpolate_na(dim="time", method="linear")
    result = filled.compute()
```

Compositing inserts empty buckets between the available observations. To retain
missing positions at either edge of a requested window, explicitly reindex onto
`samples.times[i]`; grouping does not manufacture raster values. Native `interpolate_na` owns interpolation methods and duration-based gap
limits. Rechunk time explicitly when interpolation needs one time chunk.
Interpolation keeps observation bounds unchanged; reconstructed pixels can
participate in later composites without inventing observed coverage. Sampling
adds no separate interpolation settings or semantics.

For forecasting, separate history from future observations before interpolation
that could use future information. Sampling does not assign model input/target
roles or establish product availability times.

Close opened trees after consuming their lazy pixels. Persist the catalog and
recreate the sampling request after reading it; `GeoSamples` is a row grouper,
not a raster container or an accessor.
