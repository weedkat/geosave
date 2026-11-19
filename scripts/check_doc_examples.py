"""Run introductory documentation examples and optionally refresh their figures.

Only plain Python fences are executable examples. Fences with a title are
adaptation snippets requiring user files. Network acquisition is opt-in.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "index.md",
    "basics/first-raster.md",
    "advanced/vegetation-zones.md",
    "basics/rasters.md",
    "basics/attrs.md",
    "basics/io.md",
    "basics/catalog.md",
    "basics/vectors.md",
    "advanced/crop-imagery.md",
    "advanced/large-data.md",
    "advanced/overlapping-chips.md",
    "advanced/models.md",
    "advanced/model-inputs.md",
    "concepts/xarray-structure.md",
    "concepts/lazy-data.md",
    "concepts/raster-grids.md",
    "advanced/release.md",
    "basics/training-chips.md",
)


def read_example(page: str) -> str:
    """Read a page's sequential, self-contained Python blocks."""
    text = (ROOT / "docs" / page).read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    if not blocks:
        raise ValueError(f"No executable Python blocks in {page}")
    return "\n".join(blocks)


def check_result(page: str, values: dict) -> None:
    """Check pixels, metadata, laziness and saved artifacts after execution."""
    import numpy as np
    import torch
    import geosave as gs

    if page == "index.md":
        assert values["ndvi"].gs.geobox == values["scene"].gs.geobox
        assert values["scene"].red.chunks is not None
        assert values["ndvi"].chunks is not None
        values["scene"].close()
    elif page == "basics/first-raster.md":
        scene = values["scene"]
        ndvi = values["ndvi"]
        assert scene.red.chunks is not None
        assert ndvi.chunks is not None
        decoded = scene.gs.mask_and_scale()
        expected = (decoded.nir - decoded.red) / (decoded.nir + decoded.red)
        with gs.open_raster("output/ndvi.tif") as saved:
            assert saved.gs.geobox == scene.gs.geobox
            # The file stores float32; compare at that precision.
            np.testing.assert_allclose(
                saved.ndvi.values, expected.values, atol=1e-6, equal_nan=True
            )
    elif page == "advanced/vegetation-zones.md":
        summary = values["summary"]
        assert summary.index.tolist() == ["West", "East"]
        assert (summary["coverage"] > 0.9).all()
        with gs.open_raster("output/vegetation-zones/ndvi.tif") as saved:
            assert saved.gs.geobox == values["scene"].gs.geobox
            assert np.isnan(saved.ndvi.gs.nodata)
    elif page == "basics/rasters.md":
        with gs.open_raster("output/ndvi.tif") as saved:
            assert saved.ndvi.shape == (337, 267)
            assert saved.gs.geobox == values["scene"].gs.geobox
    elif page == "basics/attrs.md":
        assert values["labels"].label.gs.flags.classes[1] == "trees"
        assert values["labels"].label.gs.nodata == 255
        assert values["labels"].label.dtype == np.uint8
        assert values["reflectance"].red.chunks is not None
        assert values["image"].red.dtype == np.uint16
        assert values["reflectance"].red.dtype == np.float64
        np.testing.assert_allclose(
            values["reflectance"].red.values,
            values["image"].red.where(values["image"].red != 0).values * 0.0001,
            equal_nan=True,
        )
    elif page == "basics/io.md":
        for name in ("scene.tif", "scene.zarr", "scene.nc"):
            with gs.open_raster(Path("output") / name) as saved:
                assert set(saved.data_vars) == {"red", "nir"}
                assert saved.gs.geobox == values["image"].gs.geobox
                for band in ("red", "nir"):
                    source = values["image"][band]
                    np.testing.assert_array_equal(saved[band].values, source.values)
                    assert saved[band].dtype == source.dtype
                    assert saved[band].gs.nodata == source.gs.nodata
                    assert saved[band].gs.packing == source.gs.packing
    elif page == "basics/vectors.md":
        from geopandas.testing import assert_geodataframe_equal

        assert_geodataframe_equal(values["reopened"], values["selected"])
    elif page == "advanced/overlapping-chips.md":
        assert len(values["chips"]) == 9
        assert values["sample"]["image"].shape == (4, 128, 128)
        assert values["sample"]["labels"].shape == (1, 128, 128)
        assert values["sample"]["labels"].dtype == torch.uint8
        assert values["chip"]["image"].gs.geobox == values["chip"]["labels"].gs.geobox
        expected = values["labels"].gs.crop_to_grid(values["chip"].gs.geobox)
        np.testing.assert_array_equal(
            values["sample"]["labels"][0].numpy(), expected.label.values
        )
    elif page == "basics/catalog.md":
        assert values["items"]["id"].tolist() == ["20250627"]
        assert values["loaded"].gs.geobox == values["scene"].gs.geobox
        assert values["loaded"].sizes["time"] == 1
        for band in ("red", "green", "blue", "nir"):
            assert values["loaded"][band].chunks is not None
            np.testing.assert_array_equal(
                values["loaded"][band].values[0],
                values["scene"][band].values,
            )
    elif page == "concepts/lazy-data.md":
        assert values["ndvi"].chunks == ((128, 128, 81), (128, 128, 11))
    elif page == "concepts/raster-grids.md":
        assert values["aligned"]
    elif page == "concepts/xarray-structure.md":
        import xarray as xr

        red = values["red"]
        scene = values["scene"]
        assert red.dims == ("y", "x")
        assert dict(red.sizes) == {"y": 2, "x": 3}
        assert values["position"] == 1
        np.testing.assert_array_equal(values["selected"].values, [0.2, 0.5])
        assert list(red.indexes) == ["y", "x"]
        assert red.column_name.dims == ("x",)
        assert values["by_name"].indexes["column_name"].get_loc("centre") == 1
        assert list(scene.data_vars) == ["red", "nir", "cloud"]
        assert scene.cloud.dims == ("x",)
        xr.testing.assert_identical(scene["red"], red)
        inherited = values["image"].to_dataset(inherit="all_coords")
        assert "time" in inherited.coords
        assert "title" not in inherited.attrs
    elif page == "advanced/model-inputs.md":
        image = values["sample"]["image"]
        assert image.shape == (2, 337, 267)
        assert image.dtype == torch.float32
        with gs.open_raster("sample-scene.tif") as source:
            expected = source[["red", "nir"]].gs.mask_and_scale().fillna(0)
            np.testing.assert_allclose(
                image.numpy(), expected.gs.to_numpy(dtype="float32")
            )
    elif page == "advanced/crop-imagery.md":
        with gs.open_raster("prepared/ndvi.tif") as saved:
            assert saved.gs.geobox == values["area"].gs.geobox
            assert saved.ndvi.shape == (169, 134)
    elif page == "advanced/large-data.md":
        assert values["ndvi"].chunks is not None
        with gs.open_raster(values["store"]) as saved:
            assert saved.gs.geobox == values["area"].gs.geobox
            np.testing.assert_allclose(
                saved.ndvi.values,
                values["ndvi"].values,
                equal_nan=True,
            )
    elif page == "advanced/release.md":
        import xarray as xr

        xr.testing.assert_identical(values["product"], values["reloaded_product"])
        with gs.open_raster("sample-scene.tif") as source:
            assert values["product"].gs.geobox == source.gs.geobox
        classes = values["product"]["class_id"]
        assert classes.dtype == np.uint8
        assert classes.gs.nodata == 255
        assert classes.gs.flags.classes == {0: "water", 1: "vegetation", 2: "bare"}
        with gs.open_raster(values["store"]) as saved:
            assert saved.gs.geobox == values["product"].gs.geobox
            assert saved.class_id.dtype == classes.dtype
            assert saved.class_id.gs.nodata == classes.gs.nodata
            assert saved.class_id.gs.flags == classes.gs.flags
            assert np.isin(saved.class_id.values, [0, 1, 2]).all()
    elif page == "basics/training-chips.md":
        sample = values["sample"]
        assert len(values["chips"]) == 4
        assert set(sample) == {"image", "labels"}
        assert tuple(sample["image"].shape) == (4, 128, 128)
        assert tuple(sample["labels"].shape) == (1, 128, 128)
        assert sample["image"].is_floating_point()
        assert sample["labels"].dtype == torch.uint8
        assert torch.isfinite(sample["image"]).all()
        with gs.open_raster("training-labels.tif") as labels:
            expected = labels.label.isel(y=slice(0, 128), x=slice(0, 128))
            np.testing.assert_array_equal(sample["labels"][0].numpy(), expected.values)
            assert values["chip"].gs.geobox == expected.gs.geobox
    elif page == "advanced/download-imagery.md":
        with gs.open_raster("imagery/scene.tif") as saved:
            assert saved.gs.geobox == values["scene"].gs.geobox
            assert set(saved.data_vars) == {"red", "green", "blue", "nir"}
            assert np.isfinite(saved.red.values).any()


def save_xarray_previews(values: dict) -> None:
    """Export native notebook displays and illustrate the example's index lookup."""
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.patches import Rectangle

    for name, obj in (("dataarray", values["red"]), ("dataset", values["scene"])):
        with xr.set_options(display_default_indexes=True, display_expand_indexes=True):
            display = obj._repr_html_()
        document = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>xarray {name} example</title>"
            "<style>body{margin:8px;background:white;color:#222;"
            "font:14px system-ui,sans-serif}.xr-wrap{max-width:100%}</style>"
            f"<body>{display}</body></html>"
        )
        (ROOT / "docs/assets" / f"xarray-{name}.html").write_text(document)

    red = values["red"]
    selected = values["position"]
    figure, ax = plt.subplots(figsize=(8, 4.5))
    ax.set(xlim=(-1.1, 3.2), ylim=(-0.65, 2.9))
    ax.axis("off")
    ax.text(1.5, 2.75, "x: 3 columns", ha="center", weight="bold", fontsize=14)
    ax.text(-0.2, 2.25, "label", ha="right", fontsize=11, color="#555555")
    ax.text(-0.2, 1.85, "position", ha="right", fontsize=11, color="#555555")
    ax.text(-0.95, 0.8, "y: 2 rows", rotation=90, va="center", fontsize=13)
    for column, label in enumerate(red.x.values):
        color = "#215ca0" if column == selected else "#333333"
        ax.text(column + 0.5, 2.25, str(label), ha="center", color=color, fontsize=14)
        ax.text(column + 0.5, 1.85, str(column), ha="center", color=color, fontsize=12)
        for row, y_label in enumerate(red.y.values):
            bottom = (1 - row) * 0.8
            ax.add_patch(
                Rectangle(
                    (column, bottom),
                    1,
                    0.8,
                    facecolor="#dcecfb" if column == selected else "#f3f5f7",
                    edgecolor="white",
                    linewidth=3,
                )
            )
            ax.text(
                column + 0.5,
                bottom + 0.4,
                f"{red.values[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=18,
                color="#222222",
            )
            if column == 0:
                ax.text(
                    -0.15,
                    bottom + 0.4,
                    str(y_label),
                    ha="right",
                    va="center",
                    fontsize=13,
                )
    ax.annotate(
        "",
        xy=(selected + 0.5, 1.64),
        xytext=(selected + 0.5, 1.78),
        arrowprops={"arrowstyle": "->", "color": "#215ca0", "lw": 2},
    )
    ax.text(
        1.5,
        -0.4,
        "sel(x=110)  =  isel(x=1)  →  [0.2, 0.5]",
        ha="center",
        fontsize=13,
        color="#215ca0",
    )
    figure.savefig(ROOT / "docs/assets/xarray-index.svg", bbox_inches="tight")
    plt.close(figure)


def save_figures(page: str, values: dict) -> None:
    """Save the actual plot objects returned by the documentation examples."""
    import holoviews as hv

    if page == "concepts/xarray-structure.md":
        save_xarray_previews(values)
        return

    plots = {
        "index.md": {"preview": "home-preview.png"},
        "basics/first-raster.md": {
            "image_plot": "first-raster-rgb.png",
            "ndvi_plot": "first-raster-ndvi.png",
        },
        "advanced/crop-imagery.md": {"ndvi_plot": "cropped-ndvi.png"},
        "advanced/overlapping-chips.md": {"chip_plot": "overlapping-chip.png"},
        "basics/training-chips.md": {
            "image_plot": "training-image.png",
            "label_plot": "training-labels.png",
        },
    }
    for name, filename in plots.get(page, {}).items():
        hv.save(
            values[name],
            ROOT / "docs/assets" / filename,
            backend="matplotlib",
            dpi=140,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Refresh static figures from the executed examples",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Also run the public Earth Search acquisition tutorial",
    )
    args = parser.parse_args()
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="geosave-docs-") as directory:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / "matplotlib"))
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        for name in ("sample-scene.tif", "training-image.tif", "training-labels.tif"):
            shutil.copy(ROOT / "docs/assets" / name, directory)
        os.chdir(directory)
        try:
            pages = list(PAGES)
            if args.network:
                pages.append("advanced/download-imagery.md")
            for page in pages:
                started = time.monotonic()
                values = {"__name__": "__main__"}
                exec(compile(read_example(page), page, "exec"), values)
                check_result(page, values)
                if args.figures:
                    save_figures(page, values)
                print(f"PASS {page} ({time.monotonic() - started:.1f}s)", flush=True)
        finally:
            os.chdir(original)


if __name__ == "__main__":
    main()
