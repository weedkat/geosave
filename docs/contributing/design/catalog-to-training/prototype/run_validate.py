# ruff: noqa
# Throwaway prototype run; see README.md.
import sys, shutil
from pathlib import Path
import geopandas as gpd, pyarrow.parquet as pq
import geosave as gs
from validate import find_problems
from geosave.geodata.stac import create_catalog

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    for name in (
        "catalog.parquet",
        "samples/catalog.parquet",
        "ard/catalog.parquet",
        "predictions/catalog.parquet",
    ):
        rows = gs.open_vector(root / name)
        print(f"{name:<30} rows={len(rows):<4} problems={find_problems(rows)}")
    raw = gs.open_vector(root / "catalog.parquet")
    broken = raw.copy()
    broken.attrs = raw.attrs
    broken.loc[1, "id"] = broken.loc[0, "id"]
    broken = broken.to_crs("EPSG:3857")
    print("broken copy:", find_problems(broken))

    # One file per collection, each with only its own assets.
    parts = root / "parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir()
    items = raw.gs.to_items()
    collections = raw.gs.collections
    for name in ("s2", "s1", "labels"):
        own = [item for item in items if item.collection_id == name]
        table = create_catalog(own, collections=[collections[name]])
        table.gs.to_geoparquet(parts / f"{name}.parquet", overwrite=True)
        print(
            name,
            "assets struct:",
            pq.read_schema(parts / f"{name}.parquet").field("assets").type.num_fields,
            "keys",
        )
    try:
        whole = gpd.read_parquet(parts)
        print("gpd.read_parquet(dir) with differing assets:", len(whole))
    except Exception as error:
        print("gpd.read_parquet(dir) fails:", type(error).__name__, str(error)[:160])
    frames = [
        gs.open_vector(parts / f"{name}.parquet") for name in ("s2", "s1", "labels")
    ]
    from geosave.geodata.core.vector.catalog import GeoCatalog

    try:
        whole = GeoCatalog.concat(frames)
        print(
            "concat of per-file frames:",
            len(whole),
            whole["assets"].dtype if hasattr(whole["assets"], "dtype") else "",
        )
        print("concat problems:", find_problems(whole))
        print("to_items after concat:", len(whole.gs.to_items()))
    except Exception as error:
        print("concat fails:", type(error).__name__, str(error)[:300])
