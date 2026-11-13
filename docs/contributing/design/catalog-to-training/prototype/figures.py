# ruff: noqa
# Throwaway prototype for docs/contributing/design/catalog-to-training.md; not library code.
"""Figures for the catalog-to-training design docs, drawn from the spike data and timings."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from odc.geo import xy_
from odc.geo.geom import Geometry
from odc.geo.gridspec import GridSpec

import geosave as gs

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def draw_map(catalog, out: Path):
    rows = catalog.to_crs("EPSG:32633")
    fig, ax = plt.subplots(figsize=(8, 7))
    styles = {"s2": BLUE, "s1": ORANGE, "labels": AQUA}
    names = {
        "s2": "S2 tile footprint (EPSG:32633)",
        "s1": "S1 footprint (native grid EPSG:32634)",
        "labels": "label footprint",
    }
    for collection, color in styles.items():
        shape = rows[rows["collection"] == collection].geometry.iloc[0]
        x, y = shape.exterior.xy
        ax.plot(
            np.array(x) / 1000,
            np.array(y) / 1000,
            color=color,
            lw=2,
            label=names[collection],
        )
    spec = GridSpec(
        "EPSG:32633", tile_shape=(1024, 1024), resolution=10, origin=xy_(0, 0)
    )
    aoi = Geometry(
        catalog[catalog["collection"] == "labels"].geometry.union_all(), crs="EPSG:4326"
    )
    for position, (index, tile) in enumerate(spec.tiles_from_geopolygon(aoi)):
        x0, y0, x1, y1 = tile.boundingbox
        ax.add_patch(
            patches.Rectangle(
                (x0 / 1000, y0 / 1000),
                (x1 - x0) / 1000,
                (y1 - y0) / 1000,
                fill=False,
                ec=INK_2,
                lw=1,
                ls="--",
                label="GridSpec tile 1024² (Design C)" if position == 0 else None,
            )
        )
        ax.text(
            x0 / 1000 + 0.2, y1 / 1000 - 0.6, f"tile {index}", color=INK_2, fontsize=8
        )
    # 256² chips with 64 px overlap over the 2048² planning grid, one row drawn.
    origin_x, origin_y = 500000, 4020480
    for column in range(0, 2048 - 255, 192):
        ax.add_patch(
            patches.Rectangle(
                ((origin_x + column * 10) / 1000, (origin_y - 2560) / 1000),
                2.56,
                2.56,
                fill=False,
                ec=MUTED,
                lw=0.8,
                label="256² chips, overlap 64 (Designs A/B)" if column == 0 else None,
            )
        )
    ax.add_patch(
        patches.Rectangle(
            (504.1, 4011.1),
            2.56,
            2.56,
            fc=BLUE,
            alpha=0.25,
            ec=BLUE,
            lw=1.5,
            label="one sample: chip GeoBox, every collection warped onto it",
        )
    )
    ax.set_xlabel("easting, km (EPSG:32633)")
    ax.set_ylabel("northing, km")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper left", bbox_to_anchor=(0, -0.1), frameon=False, ncol=1, fontsize=9
    )
    ax.set_title(
        "Spike data: three collections on three grids, and the grids samples are cut on",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_timeline(catalog, out: Path):
    s2 = (
        catalog[catalog["collection"] == "s2"]
        .sort_values("datetime")["datetime"]
        .dt.tz_convert(None)
    )
    s1 = (
        catalog[catalog["collection"] == "s1"]
        .sort_values("datetime")["datetime"]
        .dt.tz_convert(None)
    )
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for month, shade in (
        (pd.Timestamp("2024-01-01"), True),
        (pd.Timestamp("2024-02-01"), False),
        (pd.Timestamp("2024-03-01"), True),
    ):
        end = month + pd.offsets.MonthBegin(1)
        if shade:
            ax.axvspan(month, end, color=GRID, alpha=0.6, lw=0)
        ax.text(
            month + pd.Timedelta(days=1),
            3.35,
            f"{month:%b} slot (Design C)",
            color=INK_2,
            fontsize=8,
        )
    ax.scatter(
        s2,
        [2] * len(s2),
        s=60,
        color=BLUE,
        zorder=3,
        label="S2 acquisition",
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax.scatter(
        s1,
        [1] * len(s1),
        s=60,
        color=ORANGE,
        zorder=3,
        label="S1 acquisition",
        edgecolor=SURFACE,
        linewidth=2,
    )
    index = pd.Index(s1.values)
    nearest = index.get_indexer(
        s2.values, method="nearest", tolerance=pd.Timedelta("4D")
    )
    for when, match in zip(s2.values, nearest):
        if match >= 0:
            ax.plot([when, s1.values[match]], [2, 1], color=MUTED, lw=1, zorder=2)
        else:
            ax.text(when, 1.55, "no S1\n≤4 d", ha="center", fontsize=7, color=INK_2)
    for number, first in enumerate(range(0, len(s2) - 3, 2)):
        start = s2.values[first]
        end = s2.values[first + 3]
        level = 2.45 + 0.13 * (number % 3)
        ax.plot([start, end], [level, level], color=INK_2, lw=2, solid_capstyle="round")
        ax.text(start, level + 0.03, f"w{number}", fontsize=7, color=INK_2)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["S1", "S2"])
    ax.set_ylim(0.6, 3.6)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        "Time: S2 windows (size 4, step 2), S1 paired by nearest within 4 days, monthly calendar slots",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def draw_throughput(out: Path):
    rows = [
        ("Remote COG per sample, 16 threads", 0.30, None),
        ("Raw COG, per sample (ODC load)", 3.7, 10.9),
        ("Raw COG, virtual cube (A)", 4.2, 11.5),
        ("ARD COG tiles, aligned (C)", 11.1, None),
        ("Region cube Zarr (A, materialized)", 16.9, None),
        ("Prepared samples, Zarr (B alt.)", 49.4, 105.4),
        ("Prepared samples, LitData (B)", 155.8, None),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for position, (label, single, workers) in enumerate(rows):
        ax.scatter(
            single,
            position,
            s=70,
            color=BLUE,
            zorder=3,
            label="1 process" if position == 0 else None,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.text(single * 1.12, position + 0.18, f"{single:g}", fontsize=8, color=INK_2)
        if workers is not None:
            ax.plot([single, workers], [position, position], color=GRID, lw=2, zorder=1)
            ax.scatter(
                workers,
                position,
                s=70,
                color=ORANGE,
                zorder=3,
                label="4 workers" if position == 1 else None,
                edgecolor=SURFACE,
                linewidth=2,
            )
            ax.text(
                workers * 1.12, position + 0.18, f"{workers:g}", fontsize=8, color=INK_2
            )
    ax.set_xscale("log")
    ax.set_xlim(0.2, 400)
    from matplotlib.ticker import FixedLocator, FuncFormatter

    ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30, 100, 300]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("samples per second (4 × S2 + 4 × S1 + label, 256², log scale)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(
        "Measured read throughput per storage form\n6-core laptop, local disk except the remote row",
        x=0.02,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    catalog = gs.open_vector(root / "catalog.parquet")
    draw_map(catalog, out / "grids.png")
    draw_timeline(catalog, out / "time.png")
    draw_throughput(out / "throughput.png")
    print("written", sorted(path.name for path in out.iterdir()))
