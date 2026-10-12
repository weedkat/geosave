# Catalog and window rules

Use the [catalogs and windows guide](../guides/catalogs-and-cuts.md) for the
runnable path from files to a model input. These rules explain what its tables
represent and how selections behave.

## Items describe observations; windows describe inputs

| Object | One row represents | Stores pixels? |
| --- | --- | --- |
| Item table | One observation with a footprint, date, and assets. | No; assets point to files. |
| Window table | A target grid and the source Items needed on it. | No; `sources` records the Items. |
| Opened stack | Named raster groups read onto the window grid. | Arrays; typically lazy until computed. |

Both tables are GeoDataFrames. Item footprints and generated window footprints
use EPSG:4326. The `proj:*` fields describe the actual pixel grid, which can be
in a projected CRS. A geographic footprint alone does not determine that grid.

## Select complete spatial coverage

`gs.cuts.stacks(items, on="label")` uses one Item from the named collection as
each anchor. Other collections contribute Items only when they cover that
anchor's whole footprint. Partial intersections warn and are omitted; the
function does not mosaic partial scenes to make coverage complete.

Every Item must name its collection. Duplicate instants within another
collection keep the first covering Item in table order and warn. An anchor's
assets must state its grid, because a footprint cannot supply exact pixel
coordinates.

## Match observations over time

`gs.cuts.frames(windows, length, tolerance=...)` constructs a joint sequence
from collections with more than one Item. These collections need dated
observations and a shared time interval. Collections with one Item are treated
as static and join every frame, including a single dated label.

At each joint instant, the nearest observation from each dynamic collection
is selected. The tolerance bounds how far it can be from that instant; equality
with the tolerance is allowed. A tie chooses the later observation. One source
observation can therefore be selected more than once. Selection retains the
source acquisition dates; it does not interpolate values onto the joint axis.

`length` counts joint instants. The default stride equals length; a smaller
stride makes overlapping frames. A trailing sequence too short to fill a frame
is dropped with a warning. The default `mode="strict"` raises if a frame contains
an instant with missing observations. `mode="drop"` omits such frames and warns.

## Cut chips and pad edges

```python
import geosave_engine as gs

tiler, halo = gs.cuts.layout((128, 128), (64, 64), overlap=16)
halo
```

Output:

```text
[(24, 24), (24, 24)]
```

`gs.cuts.chips` uses this layout to cut windows into fixed pixel shapes. With
overlap, a halo puts the original border away from chip edges so tapered
predictions can cover it. This can produce more chips than a simple division
of the unpadded width and height suggests.

Parts of a chip outside its parent are padded using `reflect`, `edge`, or
`constant`, according to `mode`. The default is `reflect`. Padding extends the
input layout; it does not discover new observations beyond the parent window.
The merge reconstructs the layout using the recorded shape, overlap, and mode.

## Interpret row fields

| Field | Meaning |
| --- | --- |
| `anchor` | Item ID that supplied the original window grid. |
| `parent` | Position of the row in the table passed to this cut. |
| `sources` | JSON text mapping collection names to member Items. |
| `start_datetime`, `end_datetime` | Span of the selected source observations. |
| `proj:code`, `proj:transform`, `proj:shape` | Target CRS, transform, and pixel shape. |
| `chip`, `halo`, `overlap`, `mode`, `pad` | Chip placement, layout, and padding. |

`parent` is a position, not a pandas index label or a global ID. Reordering a
parent table after cutting breaks that positional relationship. Preserve the
tables used to construct predictions and pass the matching tables to merge.

## Read and persist windows

`row.gs.to_stack()` opens the member files, selects a same-grid region, and
reprojects a different grid onto the window's grid. The output is a file-backed
DataTree with lazy arrays. Two sources providing the same variable at one
instant raise rather than being silently fused. Close the stack when finished.

Item-table writes make asset hrefs relative to their catalog; reads resolve
them back to absolute addresses. Window sources contain absolute addresses
inside JSON text. Saving and reopening a window table preserves those
addresses, so it works in another process only while the sources remain
accessible there. Remote source files require the reader's credentials as
well as access to the table.

See the [cuts reference](../reference/cuts.md) for layout, reading, and merging
options.
