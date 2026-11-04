"""Cross-CRS timing and comparison with a whole-array Rasterio reference."""

import json
from pathlib import Path
import dask
import numpy as np
import rasterio.warp
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
import geosave as gs  # noqa: F401
from benchmark import measure, assert_grid


def main():
    dask.config.set(scheduler="threads", num_workers=4)
    pixels = np.random.default_rng(7).integers(
        1, 10000, (4, 1024, 1024), dtype="uint16"
    )
    grid = GeoBox((1024, 1024), Affine(10, 0, 500000, 0, -10, 5010240), "EPSG:32633")
    target = grid.to_crs("EPSG:3857", resolution=10)
    variables = {}
    for i, n in enumerate(["red", "green", "blue", "nir"]):
        variables[n] = (("y", "x"), pixels[i])
    scene = (
        xr.Dataset(variables)
        .gs.write_grid(grid)
        .gs.write_nodata(0)
        .chunk({"y": 256, "x": 256})
    )
    rows = []
    differences = []
    for kernel in ["nearest", "bilinear"]:
        measure(
            kernel + ".plan",
            lambda: scene.gs.reproject(target, resampling=kernel),
            lambda r: assert_grid(r, target),
            rows,
            repeats=3,
            tasks=True,
        )
        warped = scene.gs.reproject(target, resampling=kernel)
        measure(
            kernel + ".compute",
            warped.compute,
            lambda r: assert_grid(r, target),
            rows,
            repeats=3,
            tasks=True,
            raw_mib=8,
        )
        output = warped.red.compute().values
        reference = np.zeros(tuple(target.shape), dtype=output.dtype)
        rasterio.warp.reproject(
            pixels[0],
            reference,
            src_transform=grid.transform,
            src_crs=str(grid.crs),
            dst_transform=target.transform,
            dst_crs=str(target.crs),
            src_nodata=0,
            dst_nodata=0,
            resampling=getattr(rasterio.enums.Resampling, kernel),
        )
        valid = np.isfinite(output) & (output != 0) & (reference != 0)
        error = np.abs(
            output[valid].astype("float64") - reference[valid].astype("float64")
        )
        eager = scene.compute().gs.reproject(target, resampling=kernel).red.values
        eager_valid = (eager != 0) & (reference != 0) & np.isfinite(eager)
        eager_error = np.abs(
            eager[eager_valid].astype("float64")
            - reference[eager_valid].astype("float64")
        )
        lazy_eager_valid = valid & eager_valid
        lazy_eager_error = np.abs(
            output[lazy_eager_valid].astype("float64")
            - eager[lazy_eager_valid].astype("float64")
        )
        differences.append(
            {
                "kernel": kernel,
                "eager_vs_reference_different_pixels": int(
                    np.count_nonzero(eager_error)
                ),
                "eager_vs_reference_max_error": float(eager_error.max()),
                "lazy_vs_eager_different_pixels": int(
                    np.count_nonzero(lazy_eager_error)
                ),
                "lazy_vs_eager_max_error": float(lazy_eager_error.max()),
                "source_shape": [1024, 1024],
                "destination_shape": list(target.shape),
                "compared_pixels": int(valid.sum()),
                "different_pixels": int(np.count_nonzero(error)),
                "max_absolute_error": float(error.max()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "nodata_mask_disagreement": int(
                    np.count_nonzero(
                        (~np.isfinite(output) | (output == 0)) != (reference == 0)
                    )
                ),
            }
        )
    Path(__file__).with_name("cross-crs-results.json").write_text(
        json.dumps({"rows": rows, "reference_comparisons": differences}, indent=2)
        + "\n"
    )
    print(json.dumps(differences), flush=True)


if __name__ == "__main__":
    main()
