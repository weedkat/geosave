# Prototype for the catalog-to-training design

Throwaway code that checked the designs. It is not library code. Every file
starts with `# ruff: noqa`, and nothing here is imported by `geosave`.

| File | What it is |
| --- | --- |
| `cube.py` | the virtual cube: `CollectionReader`, `CatalogArray`, `open_cube`, `open_tree`, `select_slots`, `coverage` |
| `plan.py` | `tile`, `read_geobox`, `window`, `to_sample_items` |
| `publish.py` | `to_catalog`: expand_stac → to_cog → Items from tree metadata |
| `validate.py` | `find_problems`: the stac-geoparquet contract checks |
| `sample_io.py` | importable sample loader for worker processes |
| `figures.py` | the PNGs on the design pages |
| `spike_*.py` | the first iteration: tree-first vs per-sample, Zarr, LitData, scale |
| `run_*.py` | one script per claim on the design pages |

Run from this folder, in this order. `DATA` is any empty folder.

```bash
uv run python spike_make_data.py $DATA 2048 12        # synthetic COGs + catalog.parquet
uv run python run_cube.py $DATA                       # design A
uv run python run_coverage.py $DATA
uv run python run_dask.py $DATA
uv run python run_zones.py $DATA
uv run python run_infer_publish.py $DATA              # writes predictions/
uv run --with litdata python run_sample_catalog.py $DATA 1000   # design B, writes samples/ (~4.7 GB at 1000)
uv run python run_ard.py $DATA                        # design C, writes ard/
uv run python run_more.py $DATA
uv run python run_validate.py $DATA                   # needs the three outputs above
uv run python figures.py $DATA ..
```

`run_sample_catalog.py` repeats the 125-sample plan to 1,000 records and
writes about 4.7 GB.
