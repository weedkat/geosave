# Troubleshooting

Start with the symptom below. For exact accepted arguments, use the
[API reference](reference/index.md); for conceptual questions, see the [FAQ](faq.md).

## Installation and versions

**An example has a missing method or unexpected argument.** Check the version
in the same environment or notebook kernel that runs your code:

```bash
python -c "from importlib.metadata import version; print(version('geosave'))"
```

These docs follow `main`. Compare with the installed release, or use the
[test build](install.md#test-build-testpypi) for current development features.
Restart a notebook kernel after changing its installed packages.

**Importing torch or Lightning fails.** Install `geosave[ml]` in that environment.
A base GeoSave installation supports geospatial processing without the training
stack. See [installation extras](install.md#choose-what-you-need).

## Files and bands

**`sample-scene.tif` cannot be found.** Download the file from
[the first tutorial](first-raster.md#open-the-sample), and save it in Python's
working directory or pass its absolute path. A notebook's working directory
may differ from the folder visible in your file browser.

**A band name raises `KeyError`.** Inspect `list(scene.data_vars)`. Names are
provider- and file-specific; `B04`, `red`, and `band_1` are not interchangeable
without knowing what the stored band represents.

**A write refuses an existing file.** Choose a new destination or explicitly
pass `overwrite=True` to replace it.

## Search and download

**No Items match.** Check longitude/latitude order, collection ID, and dates.
Start with a wider time interval and fewer filters. A successful catalog search
does not mean every matching scene is clear over your area.

**Searching works, but pixel reads return 401 or 403.** Catalog access and asset
access may use different credentials. For Planetary Computer, pass
`patch_url=client.patch_url` into `stac.load_dataset`. See
[remote storage](concepts/storage.md#use-remote-storage) for other backends.

**A request is slow or fails.** First retry a small area, one observation, and
only the required bands. Check the provider's availability and your network.
The [download tutorial](guides/download-imagery.md) makes pixel reads explicit;
plotting and saving can trigger network requests after search has completed.

## Grids, values, and memory

**Imagery and labels appear shifted.** Compare their CRS and exact pixel grids,
not just array sizes. [Align them explicitly](concepts/raster-grids.md), with
nearest-neighbor resampling for categorical labels.

**Values look unexpectedly large or contain nodata codes.** Opening does not
automatically turn stored integers into physical values. Use
`.gs.mask_and_scale()` when appropriate and check the source's units and
metadata. Tensor conversion alone does not decode packing.

**An operation uses too much memory.** Select bands and spatial windows before
`.compute()`, plotting, saving, or tensor conversion. See
[large rasters](guides/large-data.md). Dask delays work; it does not make an
unbounded computation small.

## Maps and training

**The explorer is blank or no longer responds.** Keep the Python process or
notebook kernel alive. Try `.serve(port=5007)` if another server holds the
default port; select `basemap="None"` to rule out basemap network access.
Remote notebook servers may need port forwarding. See
[explorer serving](concepts/explorer.md).

**A generated starter fails on the image shape.** Inspect one Dataset sample
and one batch. DINOv3 expects `(batch, band, y, x)`; a retained time dimension
needs a temporal model or an explicit observation selection. See
[current starter checks](guides/workspaces.md#check-a-starter-before-a-long-run).

**Cross-entropy rejects the target.** A segmentation target should contain
integer class IDs, normally shaped `(batch, y, x)`, with values in
`0 … num_classes - 1` or the configured ignore code. Check that labels were
resampled with nearest-neighbor and converted to `torch.long`.

**Validation looks suspiciously good.** Check that chips from the same or
nearby scenes have not leaked across splits. Separate scenes or geographic
regions before chipping, and reserve an independent test set.

Still stuck? Include the installed version, a small reproducer, input shapes,
CRS, and the full error in a [GitHub issue](https://github.com/weedkat/geosave/issues).
Do not include credentials or private imagery.
