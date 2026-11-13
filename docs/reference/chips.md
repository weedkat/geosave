# Chips and prediction merging

`gs.Chips` accepts prepared Datasets or DataArrays. Indexing returns lazy raster
windows; `merge` accepts NumPy predictions and their keys, and returns full
float32 output arrays in source order. Merge output memory is allocated eagerly.

See [read overlapping chips and merge predictions](../guides/chips.md) for a
complete offline example, including how to place outputs on the source grid.

::: geosave.geodata.core.raster.chips.Chips
    options:
      members: [__init__, __len__, key, geobox, __getitem__, __iter__, merge]
