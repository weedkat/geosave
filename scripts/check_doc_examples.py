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
    "first-raster.md",
    "guides/ingest-files.md",
    "quickstart.md",
    "guides/training.md",
)


def read_example(page: str) -> str:
    """Read a page's sequential, self-contained Python blocks."""
    text = (ROOT / "docs" / page).read_text()
    blocks = re.findall(r"^```python\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    if not blocks:
        raise ValueError(f"No executable Python blocks in {page}")
    return "\n".join(blocks)


def check_result(page: str, values: dict) -> None:
    """Check geographic artifacts and training behavior, beyond successful execution."""
    import numpy as np
    import torch
    import geosave as gs

    if page == "first-raster.md":
        scene = values["scene"]
        ndvi = values["ndvi"]
        assert scene.red.chunks is not None
        assert ndvi.chunks is not None
        decoded = scene.gs.mask_and_scale()
        expected = (decoded.nir - decoded.red) / (decoded.nir + decoded.red)
        with gs.open_raster("output/ndvi.tif") as saved:
            assert saved.gs.geobox == scene.gs.geobox
            np.testing.assert_allclose(
                saved.ndvi.values, expected.values, equal_nan=True
            )
    elif page == "guides/ingest-files.md":
        with gs.open_raster("prepared/ndvi.tif") as saved:
            assert saved.gs.geobox == values["patch"].gs.geobox
            assert saved.ndvi.shape == (128, 128)
    elif page == "quickstart.md":
        assert len(values["chips"]) == 16
        assert isinstance(values["chip"], np.ndarray)
        assert tuple(values["image"].shape) == (2, 32, 32)
        assert tuple(values["target"].shape) == (32, 32)
        assert set(values["target"].unique().tolist()) == {0, 1}
    elif page == "guides/training.md":
        valid = values["valid"]
        image, _ = valid[0]
        with torch.inference_mode():
            before = values["model"](image.unsqueeze(0))
            after = values["restored"](image.unsqueeze(0))
        torch.testing.assert_close(before, after)
        assert torch.isfinite(before).all()
        assert torch.isfinite(values["trainer"].callback_metrics["val_loss"])
        with gs.open_raster("output/prediction.tif") as saved:
            assert saved.gs.geobox == values["scene"].isel(x=slice(64, 128)).gs.geobox
            np.testing.assert_array_equal(
                saved.prediction.values, values["prediction"].values
            )
            assert saved.prediction.dtype == np.uint8
    elif page == "guides/download-imagery.md":
        with gs.open_raster("imagery/scene.tif") as saved:
            assert saved.gs.geobox == values["scene"].gs.geobox
            assert set(saved.data_vars) == {"red", "green", "blue", "nir"}
            assert np.isfinite(saved.red.values).any()


def save_figures(values: dict) -> None:
    """Render actual tutorial data and predictions as compact documentation images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    import geosave as gs

    assets = ROOT / "docs" / "assets"
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    with gs.open_raster(assets / "sample-scene.tif") as scene:
        rgb = (
            scene[["red", "green", "blue"]]
            .to_array()
            .transpose("y", "x", "variable")
            .values
        )
        low, high = np.percentile(rgb, [2, 98], axis=(0, 1))
        rgb = np.clip((rgb - low) / (high - low), 0, 1)
        ndvi = gs.features.ndvi(scene, nir="nir", red="red").values
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 · Natural color", loc="left", fontweight="bold")
    plot = axes[1].imshow(ndvi, cmap="YlGn", vmin=-0.2, vmax=0.9)
    axes[1].set_title("Vegetation index · NDVI", loc="left", fontweight="bold")
    fig.colorbar(plot, ax=axes[1], shrink=0.72, label="NDVI")
    for axis in axes:
        axis.set_axis_off()
    fig.savefig(assets / "vegetation.jpg", dpi=140, facecolor="#ffffff")
    plt.close(fig)

    scene = values["scene"]
    colors = ListedColormap(["#263b50", "#65b98b"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    axes[0].imshow(scene.nir, cmap="YlGn", vmin=0.15, vmax=0.75)
    axes[1].imshow(scene.label, cmap=colors, vmin=0, vmax=1)
    for axis, title in zip(
        axes,
        ("Synthetic near-infrared band", "Labels · background / field"),
        strict=True,
    ):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks([31.5, 63.5, 95.5], labels=[])
        axis.set_yticks([31.5, 63.5, 95.5], labels=[])
        axis.grid(color="white", linewidth=1.2)
        axis.tick_params(length=0)
    fig.savefig(assets / "training-chips.png", dpi=130, facecolor="#ffffff")
    plt.close(fig)

    region = scene.isel(x=slice(64, 128))
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.8), layout="constrained")
    axes[0].imshow(region.nir, cmap="YlGn", vmin=0.15, vmax=0.75)
    axes[1].imshow(region.label, cmap=colors, vmin=0, vmax=1)
    axes[2].imshow(values["prediction"], cmap=colors, vmin=0, vmax=1)
    for axis, title in zip(
        axes, ("Held-out input", "Reference labels", "Model prediction"), strict=True
    ):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_axis_off()
    fig.savefig(assets / "training-result.png", dpi=130, facecolor="#ffffff")
    plt.close(fig)


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
        shutil.copy(ROOT / "docs/assets/sample-scene.tif", directory)
        os.chdir(directory)
        try:
            pages = list(PAGES)
            if args.network:
                pages.append("guides/download-imagery.md")
            for page in pages:
                started = time.monotonic()
                values = {"__name__": "__main__"}
                exec(compile(read_example(page), page, "exec"), values)
                check_result(page, values)
                if args.figures and page == "guides/training.md":
                    save_figures(values)
                print(f"PASS {page} ({time.monotonic() - started:.1f}s)", flush=True)
        finally:
            os.chdir(original)


if __name__ == "__main__":
    main()
