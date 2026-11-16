# Documentation sample data and figures

## Sentinel-2 sample

[sample-scene.tif](sample-scene.tif) contains a small crop near Davis, California:

- Source Item: `S2C_10SEH_20250627_0_L2A`, acquired June 27, 2025.
- Catalog: [Element 84 Earth Search](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2C_10SEH_20250627_0_L2A).
- Requested bounds: `(-121.82, 38.52, -121.79, 38.55)` in EPSG:4326.
- Output: EPSG:32610 at 10 metres, snapped to a 337 × 267 pixel grid.
- Bands: `red`, `green`, `blue`, `nir`, with reflectance scale `0.0001` and remaining offset `0`.
- Processing: read the selected bands onto the output grid and save as a COG.
- Radiometry: this Item declares `earthsearch:boa_offset_applied=true` while its
  asset metadata still declares offset `-0.1`. The COG conversion already
  applied that offset. The bundled TIFF records remaining offset `0` to avoid
  subtracting it twice; pixel values are unchanged. See the provider's
  [offset discussion](https://github.com/Element84/earth-search/issues/23) and
  [metadata issue](https://github.com/Element84/earth-search/issues/71).

Contains modified Copernicus Sentinel data (2025). See the
[Copernicus Sentinel data terms](https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data).
The bundled crop lets the first tutorial run without a live STAC service after
one small download. Its date and source are fixed; it is not current imagery.

`vegetation.jpg` is a natural-color rendering beside NDVI calculated from this
file. `explorer.png` shows the running GeoSave viewer with the same file and no
basemap. Color stretches are for display only.

## Training pair

[training-image.tif](training-image.tif) and
[training-labels.tif](training-labels.tif) are aligned 256 × 256 crops of
`dw_-100.2161543359_20.1295348778-20190130.tif` from the teaching dataset's
training split. Both use the first 256 rows and columns of the same tile.
The grid is EPSG:32614 at 10 metres.

- Imagery: B04/B03/B02/B08 renamed and ordered as `red`, `green`, `blue`, `nir`.
  Stored uint16 values retain scale `0.0001`, offset `0` and nodata `0`.
- Labels: expert reference classes from `classes.yaml`, stored as uint8,
  with codes 0–9 and nodata `255`. Class names and colors are embedded.
- Source snapshot: `fatmur/dynamicworld-300@5e65026ad47e1f02f9a48a3efee1028db3bb0d1a`.
- Source: Tait et al. (2021),
  [Dynamic World training dataset](https://doi.org/10.1594/PANGAEA.933475),
  [teaching subset](https://huggingface.co/datasets/fatmur/dynamicworld-300),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Contains modified Copernicus Sentinel data (2019). These are expert annotations,
not model predictions. The teaching subset remaps the source annotation codes
into its own zero-based class scheme.

## Tutorial figures

`home-preview.png`, `first-raster-rgb.png`, `first-raster-ndvi.png`,
`cropped-ndvi.png`, `training-image.png`, `training-labels.png` and
`overlapping-chip.png` are saved from the actual GeoSave plot objects in the
guides. Reproduce them from a source checkout:

```bash
uv run python scripts/check_doc_examples.py --figures
```

`explorer.png` and `explorer-comparison.png` are browser captures of the
running viewer from [map exploration](../basics/explore.md). Keep the controls
consistent with the running viewer when refreshing screenshots.

## Xarray object displays

`xarray-dataarray.html` and `xarray-dataset.html` are the native notebook
displays of the objects in [xarray structure](../concepts/xarray-structure.md).
Their expandable sections work without a Python server. `xarray-index.svg`
illustrates the same array's coordinate-to-position lookup. Refresh all three
with the `--figures` command above.

## Zone comparison

`vegetation-zones.png` shows the same sample with the two zones of
[the zone comparison](../advanced/vegetation-zones.md). Its polygons are
illustrative image halves, not field boundaries. It was rendered by a script
that is not in the repository; redraw it from that guide's `ndvi` and `zones`
if the example changes.

## Real land-cover labels

`land-cover-labels.png` shows the RGB bands and expert reference labels for
`dw_-100.2161543359_20.1295348778-20190130.tif` in the teaching dataset's
training split, snapshot `5e65026ad47e1f02f9a48a3efee1028db3bb0d1a`.
RGB uses B04/B03/B02, decoded reflectance and a 0–0.3 display stretch;
labels use `classes.yaml` colors with nodata 255 masked. This is reference
data, not model output. Source: Tait et al. (2021),
[Dynamic World training dataset](https://doi.org/10.1594/PANGAEA.933475),
[teaching subset](https://huggingface.co/datasets/fatmur/dynamicworld-300),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Contains modified Copernicus Sentinel data (2019).
