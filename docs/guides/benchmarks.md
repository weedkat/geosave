# Download benchmark labels

Download the original Dynamic World expert labels:

<!-- NETWORK: original publisher download -->
```python
from geosave_engine.geodata.benchmarks import dynamic_world

labels = dynamic_world.download("data/dynamic_world", subset="experts")
labels
```

Expected return after a successful download, rather than a captured download
result:

```text
PosixPath('data/dynamic_world/experts')
```

`labels` is a local directory containing the publisher's extracted files and
metadata. The downloader preserves their folder structure and label values.
It does not acquire source imagery or prepare a training catalog.

## Choose a subset

| Subset | Contents |
| --- | --- |
| `experts` | Expert training labels and metadata. |
| `non_expert` | Commissioned training labels and metadata. |
| `validation` | Individual expert labels, composites, and README. |
| `test` | Expert consensus labels, predictions, probabilities, and CSV metadata. |

The training and validation release contains label GeoTIFFs and identifies the
source Sentinel-2 scenes in metadata; the original imagery is not included.
It is published under CC BY 4.0. Use the publisher's citation and required
attribution. [PANGAEA training release](https://doi.pangaea.de/10.1594/PANGAEA.933475).

## Open the original rasters

After the download completes, use the normal raster reader. This block depends
on the downloaded files, so its output is not captured here:

<!-- NETWORK: requires the completed download above -->
```python
import geosave_engine as gs

files = sorted(labels.rglob("*.tif"))
label = gs.open_raster(files[0])
label
```

Use the publisher's metadata to interpret band values and match labels to
imagery. Training labels use `0` for unmarked pixels, so decide how to exclude
those pixels when preparing training targets. Avoid assuming every release
uses the same band names or class indexing.
[PANGAEA label description](https://doi.pangaea.de/10.1594/PANGAEA.933475).

For the test release, `label_*` rasters contain `lulc` for human annotation and
`label` for the model's predicted class; probabilities are in separate rasters.
Select the human annotation as ground truth. The approximately 3.9 GB test
archive has a published MD5 checksum, which this downloader verifies.
[Zenodo test release](https://zenodo.org/records/4766508).

Once you have paired imagery and labels, [catalogs and windows](catalogs-and-cuts.md)
show how to describe saved rasters as Items and select model inputs. Downloading
alone does not create that catalog or a training/validation split.

## Reuse completed downloads

<!-- NETWORK: fetches only if the subset is not complete -->
```python
labels = dynamic_world.download("data/dynamic_world", subset="experts")
```

Each subset has a `.geosave-complete.json` record. A matching completed
extraction is reused without network requests. The record confirms the
extraction completed; it does not repair files later changed or deleted.

Transfers and extraction use a temporary directory. A failed attempt does not
publish a completed subset, and temporary ZIP files are removed afterward.
Allow disk space for both the archive and extracted files during the download.

An existing subset directory without a matching completion record raises
`FileExistsError`. Move or remove that incomplete directory explicitly before
retrying. Malformed completion JSON raises `ValueError`. Downloads are local
and do not coordinate concurrent calls to the same destination.

See the [benchmark reference](../reference/benchmarks.md) for the download API.
