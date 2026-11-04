# GeoSave

GeoSave turns catalogued geospatial data into the inputs a model reads, from item tables through to lazy rasters.

## Language

### Sampling

**Region**:
One place a sample covers: founded by the smallest footprint, joined by every Item covering most of it; a large footprint, such as a radar swath, joins several.
_Avoid_: tile, area, place

**Pair**:
An Item of the first paired collection with the nearest Item of each other paired collection within the tolerance, taken as one time step dated by the first; with one collection it is that Item alone.
_Avoid_: match, group, unit, step

**Tolerance**:
How far apart, either way, a Pair's Items may lie.
_Avoid_: offset, horizon, lag

**Slot**:
One time step of a window, holding one Pair. With a frequency it is a span, such as a calendar month, filled by its latest Pair unless the earliest is asked for; without one it is simply the next Pair.
_Avoid_: step, bin, period

**Window**:
Consecutive slots of one place taken together as one sample, as pandas `rolling` takes consecutive rows.
_Avoid_: frame, lookback, sequence

**Chip**:
One fixed-size pixel window of a sample's raster, at the model's input size and pixel size; chips of one sample are cut from, and merged back onto, one layout.
_Avoid_: tile, patch, crop

**Composite**:
One Item made from several acquisitions of one collection over one slot's span, such as a monthly median.
_Avoid_: mosaic, aggregate
