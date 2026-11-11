# Calendar sampling

> Ownership correction: [2026-11-06 sampling ownership](../plans/2026-11-06-sampling-ownership.md)
> supersedes the preparation settings and GeoSamples loader described below.
> The interval assignment and calendar grouping contracts remain in effect.


Approved direction: one public `catalog.gs.sample` request, small domain operations,
no reference collection, no input/target roles or signed offsets. `window` counts
consecutive slots; data determines the timeline and `step` advances along it.

API: collections=None, match='overlap', overlap=.9, freq=None, window=1,
step=1, interval='raise', reduce='last', interpolate=None, max_gap=None,
missing='drop'. `pair`, `tolerance`, and `keep` are removed.

With all temporal options at their defaults, retain spatial-only sampling.
Explicit collections or nondefault temporal/preparation options request temporal sampling. `freq=None`
uses exact representative timestamps, and requires representative datetime for
interval-only Items. Static Items never create timestamps. Collection selection
is exact: unselected static collections are not implicitly attached.

Temporal metadata is naive UTC internally. datetime assigns an Item; its bounds
remain coverage. Interval-only Items use inclusive bounds; calendar buckets are
left-closed/right-open with start labels. Crossing policies: raise (default),
first, last, overlap, drop. Invalid or one-sided coverage raises. All-time sentinel
is static. Calendar boundaries use pandas resample with closed/label='left';
empty slots remain between the first and last assignment. No source rows mutate.

Spatial grouping supports existing overlap behavior and geometric equality.
Founding footprint is retained rather than recomputed from surviving rows.

GeoSamples retains original catalog positions, slot membership and windows.
Indexing returns source rows. `to_datatree(position)` opens and prepares one sample;
reduction is per collection and slot on native compatible grids. First/last
select complete observations, never mix pixels from different scenes. Other
reducers use existing temporal.composite. Preparation remains lazy. Missing slots
are explicit NaN values. Interpolation uses existing temporal.interpolate but
masks extrapolation and runs longer than max_gap (positive integer slot count).
Planning completeness concerns observations only, not pixel validity.
Interpolation uses only the selected window and is not forecasting-safe by itself.
A collection absent throughout a kept window has no grid to fabricate; its missing
membership remains visible, and its tree group is empty.

Tests: independent month assignments, boundary crossing policies, nominal time
precedence/coverage, invalid bounds, gaps/window steps, static members, exact
footprints, reserved collection names, non-RangeIndex positions, lazy numerical
preparation and resource closure. No compatibility aliases or notebook edits.
