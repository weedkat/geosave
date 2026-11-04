"""Compare reader/layout choices on the same COG; no network or production edits."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import dask
import numpy as np
import rasterio
import rioxarray
import torch
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
import geosave as gs
from benchmark import measure


def main():
    torch.set_num_threads(1)
    dask.config.set(scheduler="threads", num_workers=4)
    pixels = np.random.default_rng(7).integers(
        1, 10000, (4, 2048, 2048), dtype="uint16"
    )
    names = ["red", "green", "blue", "nir"]
    variables = {}
    for i, n in enumerate(names):
        variables[n] = (("y", "x"), pixels[i])
    grid = GeoBox((2048, 2048), Affine(10, 0, 500000, 0, -10, 5020480), "EPSG:32633")
    scene = xr.Dataset(variables).gs.write_grid(grid).gs.write_nodata(0)
    rows = []
    with TemporaryDirectory(prefix="gs-perf-cog-") as folder:
        path = scene.gs.to_cog(folder, path_format="scene.tif")[0]
        for chunks in [None, {}, {"y": 1024, "x": 1024}, {"y": 2048, "x": 2048}]:

            def read():
                with gs.open_raster(path, chunks=chunks) as source:
                    return source[names].gs.to_tensor().numpy()

            measure(
                f"geosave.chunks={chunks}",
                read,
                lambda a: np.testing.assert_array_equal(a, pixels),
                rows,
                raw_mib=32,
                tasks=True,
            )
        for split in [False, True]:
            for chunks in [None, {}]:

                def read():
                    with rioxarray.open_rasterio(
                        path,
                        band_as_variable=split,
                        chunks=chunks,
                        mask_and_scale=False,
                    ) as source:
                        if split:
                            return source.to_array().values
                        return source.values

                measure(
                    f"rioxarray.split={split}.chunks={chunks}",
                    read,
                    lambda a: np.testing.assert_array_equal(a, pixels),
                    rows,
                    raw_mib=32,
                    tasks=True,
                )
        individual = scene.gs.to_cog(Path(folder) / "bands")
        by_name = {p.stem: p for p in individual}

        def separate():
            arrays = []
            for n in names:
                with gs.open_raster(by_name[n]) as source:
                    arrays.append(source[n].values)
            return np.stack(arrays)

        measure(
            "geosave.separate_band_files",
            separate,
            lambda a: np.testing.assert_array_equal(a, pixels),
            rows,
            raw_mib=32,
            tasks=True,
        )
        with rasterio.open(path) as source:
            metadata = {
                "block_shapes": source.block_shapes,
                "compression": str(source.compression),
                "interleaving": str(source.interleaving),
                "file_bytes": Path(path).stat().st_size,
            }
    Path(__file__).with_name("cog-results.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
