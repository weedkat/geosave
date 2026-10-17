# Catalog series, joins and frames

Approved direction: group observations into series, join collections on an anchor
series, then select consecutive frames. Implement inline in the shared checkout;
preserve existing unrelated edits and do not modify notebooks.

```python
series_rows = items.gs.group_series()
matched_rows = series_rows.gs.join_collections(on="optical", tolerance="10D", mode="drop")
frame_rows = matched_rows.gs.select_frames(4, stride=2)
```

## Row contract

A source Item describes an observation. A series row describes one collection
and normalized footprint, with ordered source Items and `times` (ISO UTC sampling
timestamps). A timeless singleton has `times=None`. Joined rows contain one
anchor series per spatial footprint and source observations selected for each
anchor timestamp. Keep acquisition dates inside the original Item dictionaries.

The anchor collection supplies the timeline. Spatial coverage is checked before
temporal matching. Dated singletons obey tolerance; timeless observations can
broadcast, and interval observations cover timestamps inside their stated range.
Nearest observations outside intervals must be within tolerance; ties choose the
later observation. Reject ambiguous same-collection duplicate instants among
covering sources. `strict` rejects missing matches; `drop` removes incomplete
sampling positions consistently across collections. No union axis, shared-period
truncation, or deduplication of adjacent matching combinations.

Frames slice existing times and corresponding source selections only. They have
no tolerance, matching, or static-inference options. One dated observation supports
one frame of length one. Preserve positional parent relationships and the existing
frame-before-chip restriction until reconstruction lineage is redesigned.

Readers reconstruct repeated selections lazily, use sampling timestamps as `time`,
and retain the source dates as `acquisition_time(time)`. Timeless groups remain
spatial arrays. Projection metadata remains a reference grid until explicit
alignment or chip selection. GeoParquet preserves times and source selections.

## Modules and migration

`series.py` owns grouping; `join.py` owns spatial and temporal matching;
`frames.py` owns sequence slicing. Rename `frames()` to `select_frames()` and
remove its temporal-join helpers. Join accepts series rows and a collection name;
the GeoAnchor overload is removed because it supplies no collection timeline.
Update accessors, model consumers, templates, docs and tests without aliases.
Preserve and improve the ASCII timelines in docstrings and the catalog guide.

## Accepted warning policy (user steering)

Default joins warn and drop incomplete sampling positions. Invalid Item rows,
ambiguous series and unusable frame rows warn and are skipped without discarding
other valid data. Strict matching remains opt-in. Invalid caller arguments and
wrong operation stages still raise. Anchor observations are preserved verbatim;
only the other collections undergo matching. ModelSpec.prepare forwards explicit
tolerance/mode and declares its prepared Collection definitions.
