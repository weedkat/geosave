"""Additional crop/chunk/catalog benchmarks; run after benchmark.py, not concurrently."""

from datetime import datetime, timedelta, UTC
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import dask
import numpy as np
import pystac
import torch
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from shapely.geometry import box, mapping

import geosave as gs
from geosave.geodata.stac import from_items
from benchmark import measure, assert_grid


def main():
    torch.set_num_threads(1)
    dask.config.set(scheduler="threads", num_workers=4)
    rows = []
    size = 2048
    pixels = np.random.default_rng(7).integers(
        1, 10000, (4, size, size), dtype="uint16"
    )
    names = ["red", "green", "blue", "nir"]
    variables = {}
    for i, name in enumerate(names):
        variables[name] = (("y", "x"), pixels[i])
    grid = GeoBox((size, size), Affine(10, 0, 500000, 0, -10, 5020480), "EPSG:32633")
    scene = xr.Dataset(variables).gs.write_grid(grid).gs.write_nodata(0)
    target = grid[:256, :256]
    measure(
        "cast.geosave.float32",
        lambda: scene.gs.to_tensor(dtype="float32"),
        lambda t: np.testing.assert_array_equal(t.numpy(), pixels),
        rows,
        raw_mib=32,
    )
    measure(
        "cast.numpy_stack_float32",
        lambda: torch.from_numpy(
            np.stack([scene[n].values for n in names], dtype="float32")
        ),
        lambda t: np.testing.assert_array_equal(t.numpy(), pixels),
        rows,
        raw_mib=32,
    )
    for chunk in [128, 256, 512, 1024]:
        lazy = scene.chunk({"y": chunk, "x": chunk})
        measure(
            f"chunk{chunk}.to_tensor",
            lazy.gs.to_tensor,
            lambda t: np.testing.assert_array_equal(t.numpy(), pixels),
            rows,
            tasks=True,
            raw_mib=32,
        )
        measure(
            f"chunk{chunk}.crop.plan",
            lambda: lazy.gs.crop(target.extent),
            lambda r: assert_grid(r, target),
            rows,
            tasks=True,
        )
        cropped = lazy.gs.crop(target.extent)
        measure(
            f"chunk{chunk}.crop.tensor",
            cropped.gs.to_tensor,
            lambda t: np.testing.assert_array_equal(t.numpy(), pixels[:, :256, :256]),
            rows,
            tasks=True,
            raw_mib=0.5,
        )
    with TemporaryDirectory(prefix="gs-perf-catalog-") as directory:
        for regions in [2, 20, 100]:
            entries = []
            for region in range(regions):
                geometry = box(region * 0.02, 0, region * 0.02 + 0.01, 0.01)
                for day in range(50):
                    entry = pystac.Item(
                        id=f"{region}-{day}",
                        geometry=mapping(geometry),
                        bbox=list(geometry.bounds),
                        datetime=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day),
                        properties={},
                        collection="optical",
                    )
                    entry.add_asset(
                        "red",
                        pystac.Asset(
                            "https://example.invalid/red.tif",
                            media_type=pystac.MediaType.COG,
                            roles=["data"],
                        ),
                    )
                    entries.append(entry)
            count = len(entries)

            def check_frame(frame):
                assert len(frame) == count

            measure(
                f"{count}.catalog.from_items",
                lambda: from_items(entries),
                check_frame,
                rows,
                repeats=3,
            )
            catalog = from_items(entries)
            measure(
                f"{count}.catalog.sample",
                lambda: catalog.gs.sample(pair=["optical"], window=3),
                lambda s: check_count(s, regions * 48),
                rows,
                repeats=3,
            )
            path = Path(directory) / f"{count}.parquet"
            measure(
                f"{count}.catalog.write",
                lambda: catalog.gs.to_geoparquet(path, overwrite=True),
                lambda p: assert_exists(p),
                rows,
                repeats=3,
            )
            measure(
                f"{count}.catalog.open",
                lambda: gs.open_vector(path),
                check_frame,
                rows,
                repeats=3,
            )
            rows.append(
                {"name": f"{count}.catalog.bytes", "bytes": path.stat().st_size}
            )
    Path(__file__).with_name("extended-results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def check_count(value, count):
    assert len(value) == count, (len(value), count)


def assert_exists(path):
    assert path.exists()


if __name__ == "__main__":
    main()
