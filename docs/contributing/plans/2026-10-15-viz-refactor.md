# Visualization refactor

Design: [Visualization responsibilities](../specs/2026-10-15-viz-responsibilities.md).

Goal: make visualization ownership, names and filtering consistent without replacing HoloViz objects or changing the accessor workflow.

## Responsibilities

- `plot.py`: static raster figures and vector outlines.
- `shapes.py`: select, group and summarize drawable geometry.
- `map.py`: Web Mercator view, framing and basemaps.
- `raster.py`: lazy raster overviews and screen-window sampling.
- `style.py`: stretches, palettes and presentation defaults.
- `explorer/layers`: data-specific rendering, inspection and selection. Vector properties use native Tabulator column filters and shared text search; catalogs add date intervals and lazy opening.
- `explorer/docks`: widgets for filtering, tables, layer lists, styling, inspection, comparison and time.
- `explorer/app.py`: layer lifecycle, selected layer, dates, comparison and layout.

## Work

1. Reproduce row-name collisions, geographic framing, filtered highlights and explorer lifecycle/choice failures with focused tests.
2. Replace cloud-cover handling with native Tabulator column filters; map visible rows to original table positions and retain catalog interval semantics.
3. Move raster sampling out of map and split placement-based tabs into style and inspection modules. Rename action functions with verbs and ambiguous local variables by content; update callers without aliases.
4. Repair layer membership/watchers, selection and lossless date/comparison choices.
5. Run visualization and accessor tests, lint and focused type checking. Document changed internal interfaces and unresolved limitations.

## Review focus

Preserve lazy raster reads, metadata, user property names and positional selections; cover duplicate names, sub-minute times, missing property values and custom geometry names. Preserve existing working-tree edits and ignore notebooks.
