# Catalog per collection: one item table file for each STAC collection

Status: Approved

## Problem

- A catalog is one `catalog.parquet` holding the Items of every group
  (`image`, `label`, ...). STAC keeps Items of one Collection together; here the
  groups share a file only because the code writes them that way.
- Replacing one group, such as new labels, rewrites every row of every group.
- A reader that wants one group reads and discards the others.

## Layout

A *catalog* is a folder. It holds one item table per Collection, named after
it, beside the folders its hrefs point to.

```text
data/prepared/
├── image.parquet          # collection "image", stac-geoparquet
├── label.parquet          # collection "label"
├── model_spec.yaml        # prepared catalogs only
└── <stack>/image.zarr, label.zarr
```

Hrefs are stored relative to the file, so the folder moves whole. In memory a
catalog is still one item table (a GeoDataFrame): `gs.to_stack()`, `cuts` and
`supervised.Dataset` are unchanged.

A smoke test confirmed this on `examples/data/dw_raw`: written per
collection and read back, ids, hrefs, geometry and CRS equal the combined
table; hrefs open after the folder moves; rewriting `label.parquet` alone
leaves the other part intact.

## What a caller and a maintainer see

```python
# caller
catalog  = spec.prepare(raw, "data/prepared", keep=["label"])   # -> Path("data/prepared")
prepared = read_vector("data/prepared")                         # one item table
rows.gs.to_stac_parquet("data/raw", collections=[image, label])      # no suffix: a catalog
rows.gs.to_stac_parquet("data/labels.parquet", collections=[label])  # a suffix: one file
supervised.DataModule(spec, train="data/train", val="data/val")
```

```python
# maintainer: two functions beside the file-level pair
def write_catalog(rows, folder, *, collections=None, overwrite=False) -> Path:
    for collection_id, part in rows.groupby("collection"):
        write(part, folder / f"{collection_id}.parquet", collections=[...], overwrite=overwrite)

def read_catalog(folder, **options) -> GeoDataFrame:
    parts = [read(path, **options) for path in sorted(folder.glob("*.parquet"))]
    return GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
```

`write` and `read` stay the file-level pair: stac-geoparquet keeps a file's
Collections in its metadata, keyed by id, so a file may hold several. They
accept that; its docstring recommends one collection per file for anything
kept. `write_catalog` is where one per file is enforced.

## Library changes

### `stac.table`

- `write_catalog(rows, folder, *, collections=None, overwrite=False)`
  - Writes `<collection id>.parquet` for each collection in `rows`, each file
    carrying only its own Collection and the extent of its rows.
  - The folder may already exist; it holds the stacks. Only the files of the
    collections in `rows` are written, so a call with label rows alone replaces
    `label.parquet` and nothing else.
  - With `overwrite=False` every target is checked before any is written.
  - Local folders only in this version.
  - Raises `ValueError` for rows that are not an item table, a row with no
    collection, a collection id that is not a plain file name (separator,
    leading dot, empty), or a row naming a collection absent from
    `collections` when `collections` is given; `FileExistsError` as above.
  - Returns the folder.
- `read_catalog(folder, **options)`
  - Reads every `*.parquet` directly in the folder, sorted by name, with the
    `read` options (`bbox`, `filters`) applied to each part, and returns one
    item table with a fresh index.
  - Raises `FileNotFoundError` when the folder holds no parquet file.
- `read_collections(path)` accepts a folder and merges its parts' Collections.

### Public spellings

- `GeoVector.to_stac` is renamed `to_stac_parquet`, with no alias: it writes a
  table of Items, and the name states the format like `to_geoparquet`. A path
  with `.parquet` or `.geoparquet` writes one file (`table.write`); a path with
  no suffix writes a catalog.
- `read_vector(source)`: a directory is read as a catalog; a file as today.

### `ModelSpec.prepare`

- Builds the Items as now, then `table.write_catalog(items, draft, collections=...)`
  instead of `catalog.parquet`. The draft rename, the stack checks and the copy
  of the spec are unchanged.
- Returns the output folder.

### `supervised.DataModule` and `Dataset`

- `train`, `val`, `test` are catalog folders.
- `_check_prepared` reads `<catalog>/model_spec.yaml` rather than
  `<catalog's parent>/model_spec.yaml`, and refuses a path that is a file with
  a message naming the folder to pass.
- `Dataset(read_vector(folder), spec)` is otherwise unchanged. A stack that
  lacks a group the spec reads is named in the error.

## Breaking changes

`prepare` no longer writes `catalog.parquet`, and the data module takes a folder
where it took that file. Catalogs written before this change are prepared again.
There is no reader for the old layout.

## Examples and docs

- `examples/data/dw_raw/catalog.parquet` becomes `label.parquet` and
  `sentinel_2_l2a.parquet`, split from the existing file with the committed
  rows and Collections.
- `01_imagery` ended by passing a list of Items to `table.write`, which raises
  (it takes a table); it now builds the table with `table.from_items` and writes
  it with `write_catalog`. `00`–`05` pass folders; the workspace README and `train.yaml` use `data/train` and
  `data/val`.
- `tests/examples/test_notebooks.py` asserts the part files instead of
  `catalog.parquet`.
- The Examples table and data-flow text of `2026-10-08-prepare-design.md` and
  `docs/guides/architecture.md` name the folder.

## Tests

- `write_catalog`: one file per collection with only its own Collection and
  extent; round trip equals the combined table in ids, hrefs, geometry, CRS and
  Collections; rewriting one collection leaves the other part byte-identical;
  existing target raises before any file is written; row with no collection,
  unsafe collection id and absent collection raise.
- `read_catalog`: options reach every part; empty folder raises; the folder
  moves with its assets and hrefs still open.
- `read_vector` and `to_stac` dispatch: folder, `.parquet` file, no-suffix path.
- `prepare`: output folder holds one part per group plus the spec; prepared
  pixels still equal `spec.preprocess` of the raw stack; a failed run leaves
  nothing behind.
- `DataModule`: a file path is refused with the folder named; a folder without
  the spec is refused as before.
- `Dataset`: a stack missing a group names the stack.
- Every notebook executes.

## Out of scope

- Remote catalog folders: listing a bucket prefix needs its own design.
  Remote single-file tables work as today.
- `StacTableClient` searching a whole catalog folder. It takes one part file,
  which is a valid table.
- Reading catalogs written before this change.
