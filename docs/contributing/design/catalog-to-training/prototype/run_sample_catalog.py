# ruff: noqa
# Throwaway prototype run; see README.md.
"""Design B end to end: plan → GeoParquet plan → LitData → sample catalog (stac-geoparquet)."""

import sys
import time
from pathlib import Path

import pandas as pd
import pystac
from odc.geo.geobox import GeoBox

import geosave as gs
from plan import tile, to_sample_items, window
from sample_io import Prepare, load_sample, open_tables


def clock(label, work):
    began = time.perf_counter()
    result = work()
    print(f"{label:<56} {time.perf_counter() - began:8.3f}s")
    return result


def make_plan(root: Path):
    catalog = gs.open_vector(root / "catalog.parquet")
    grid = GeoBox.from_bbox(
        (500000, 4000000, 520480, 4020480), crs="EPSG:32633", resolution=10
    )
    chips = clock(
        "tile 2048² grid into 256² chips, overlap 64",
        lambda: tile(grid, 256, overlap=64),
    )
    found = chips.sjoin(
        catalog[["id", "collection", "datetime", "geometry"]], predicate="intersects"
    )
    inside = []
    for _, row in found.iterrows():
        footprint = catalog.loc[row["index_right"], "geometry"]
        inside.append(
            row["geometry"].intersection(footprint).area / row["geometry"].area
        )
    found["coverage"] = inside
    found = found[found["coverage"] > 0.99]
    s2 = found[found["collection"] == "s2"].sort_values("datetime")
    s1 = found[found["collection"] == "s1"].sort_values("datetime")
    slots = pd.merge_asof(
        s2[
            [
                "chip",
                "id",
                "datetime",
                "proj:code",
                "proj:shape",
                "proj:transform",
                "geometry",
            ]
        ].rename(columns={"id": "s2"}),
        s1[["chip", "id", "datetime"]].rename(
            columns={"id": "s1", "datetime": "s1_datetime"}
        ),
        left_on="datetime",
        right_on="s1_datetime",
        by="chip",
        direction="nearest",
        tolerance=pd.Timedelta("4D"),
    )
    labels = found[found["collection"] == "labels"][["chip", "id"]].rename(
        columns={"id": "labels"}
    )
    slots = slots.merge(labels, on="chip")
    samples = clock(
        "window 4 slots, step 2",
        lambda: window(
            slots, by="chip", size=4, step=2, collections=["s2", "s1", "labels"]
        ),
    )
    unpaired = 0
    reused = 0
    for ids in samples["s1"]:
        if "" in ids:
            unpaired += 1
        if len(set(ids)) < len(ids):
            reused += 1
    print(
        f"{len(chips)} chips → {len(samples)} samples; {unpaired} with an unpaired S1 slot, {reused} reuse an S1 Item"
    )
    path = root / "samples/plan.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(path)


if __name__ == "__main__":
    from litdata import StreamingDataLoader, StreamingDataset, optimize

    root = Path(sys.argv[1]).resolve()
    make_plan(root)
    catalog, plan = open_tables(str(root))
    print(
        plan[["id", "start_datetime", "end_datetime", "s2", "s1"]].head(2).to_string()
    )
    arrays = clock(
        "load one sample (3 collections on its grid)",
        lambda: load_sample(catalog, plan.iloc[0]),
    )
    print(
        {key: value.shape for key, value in arrays.items()},
        "s1 valid:",
        arrays["s1_valid"],
    )

    count = (
        int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    )  # plan repeated to a realistic store size
    began = time.perf_counter()
    optimize(
        fn=Prepare(str(root), count),
        inputs=list(range(count)),
        output_dir=str(root / "samples/lit"),
        chunk_bytes="128MB",
        num_workers=5,
        mode="overwrite",
    )
    print(
        f"LitData optimize {count} samples, 5 workers: {time.perf_counter() - began:.1f}s"
    )
    for workers in (0, 2, 4):
        dataset = StreamingDataset(str(root / "samples/lit"), shuffle=True)
        loader = StreamingDataLoader(dataset, batch_size=16, num_workers=workers)
        began = time.perf_counter()
        seen = 0
        for batch in loader:
            seen += batch["s2"].shape[0]
        print(
            f"LitData read {seen}, workers={workers}: {seen / (time.perf_counter() - began):6.1f} samples/s",
            tuple(batch["s2"].shape),
        )

    items = to_sample_items(plan, ["s2", "s1", "labels"], "samples-v1")
    for position, item in enumerate(items):
        item.add_asset(
            "data",
            pystac.Asset(
                (root / "samples/lit").as_uri(),
                media_type="application/vnd.litdata",
                roles=["data"],
                extra_fields={"litdata:index": position},
            ),
        )
    definition = pystac.Collection(
        "samples-v1",
        "4-slot S2 windows, nearest S1 within 4 days, landcover; 256 px chips at 10 m",
        pystac.Extent(
            pystac.SpatialExtent([[-180, -90, 180, 90]]),
            pystac.TemporalExtent([[None, None]]),
        ),
    )
    from geosave.geodata.stac import create_catalog

    table = create_catalog(items, collections=[definition])
    path = table.gs.to_geoparquet(root / "samples/catalog.parquet", overwrite=True)
    import rustac

    print(
        "sample catalog:",
        len(gs.open_vector(path)),
        "rows; rustac reads",
        len(rustac.read_sync(str(path))["features"]),
    )
    print(gs.open_vector(path)[["id", "s2", "s1", "labels"]].head(2).to_string())
