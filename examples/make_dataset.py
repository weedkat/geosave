"""Build the example dataset: Dynamic World expert labels with Sentinel-2 imagery.

Picks well-labelled, cloud-free expert tiles across every biome, loads the
Sentinel-2 scene each was drawn on from Planetary Computer, and writes

    <out>/<split>/images/<id>.tif   B02, B03, B04, B08 as uint16, scale 1e-4
    <out>/<split>/labels/<id>.tif   classes 0-9, 255 meaning unlabelled
    <out>/classes.yaml               each class's name and colour

with train, val and test holding 4:1:1 of the tiles, each split spanning the
same biomes.

    uv run python examples/make_dataset.py workspace/data/dynamicworld_raw \
        examples/data/dynamic_world --tiles 300
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import geosave_engine as gs
from geosave_engine.geodata.stac import StacClient
from geosave_engine.geodata.io import COGWriteOptions

BANDS = ["B02", "B03", "B04", "B08"]
# Dynamic World's codes 1-10, renumbered from 0 as segmentation datasets do;
# its code 0, pixels a labeller left unmarked, becomes the ignore index 255.
CLASSES = [
    ("water", "#419bdf"),
    ("trees", "#397d49"),
    ("grass", "#88b053"),
    ("flooded_vegetation", "#7a87c6"),
    ("crops", "#e49635"),
    ("scrub", "#dfc35a"),
    ("built_area", "#c4281b"),
    ("bare_ground", "#a59b8f"),
    ("snow_ice", "#b39fe1"),
    ("cloud", "#d3d3d3"),
]
UNLABELLED = 255
SPLITS = ["train"] * 4 + ["val", "test"]
# Neighbouring pixels are close in value, so differencing them first compresses
# 16-bit imagery a third smaller, and every GIS reads deflate.
COMPRESSION: COGWriteOptions = {"compress": "deflate", "predictor": 2, "zlevel": 9}


def candidates(raw: Path, tiles: int, seed: int) -> pd.DataFrame:
    """Pick well-labelled, cloud-free expert tiles, spread over biomes."""
    table = pd.read_excel(raw / "v1_dw_tile_metadata_for_public_release.xlsx")
    paths = {path.stem: path for path in (raw / "train/Experts").rglob("*.tif")}
    # Some tiles were labelled twice; the expert row describes the expert file.
    table = table[(table["labeler"] == "Experts") & table["dw_id"].isin(paths)]
    table = table[(table["NB_Pixels"] >= 0.9 * 510 * 510) & (table["Clouds"] <= 1)]
    # A few spare tiles per biome cover scenes Planetary Computer lacks.
    per_biome = -(-tiles * 6 // 5 // table["biome"].nunique())
    shuffled = table.sample(frac=1, random_state=seed)
    picked = shuffled.groupby("biome").head(per_biome)
    return picked.assign(path=picked["dw_id"].map(paths))


def read_label(path: Path):
    label = gs.open_raster(path).rename({"band_1": "label"}).compute()
    codes = label["label"].values
    label["label"].values = np.where(codes == 0, UNLABELLED, codes - 1).astype("uint8")
    names = {index: name for index, (name, _) in enumerate(CLASSES)}
    colors = {index: color for index, (_, color) in enumerate(CLASSES)}
    label = label.gs.write_nodata(UNLABELLED)
    return label.gs.write_legend(
        {**names, UNLABELLED: "unlabelled"},
        {**colors, UNLABELLED: "#000000"},
        target="label",
    )


def write_classes(out: Path) -> None:
    """Write the class table, for readers that do not open the files' legends."""
    table = {
        "ignore_index": UNLABELLED,
        "class_map": {index: name for index, (name, _) in enumerate(CLASSES)},
        "color_map": {index: color for index, (_, color) in enumerate(CLASSES)},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.yaml").write_text(yaml.safe_dump(table, sort_keys=False))


def load_image(source, label):
    image = source.load(label.gs.anchor)[BANDS].compute()
    image = image.fillna(0).astype("uint16").gs.write_nodata(0)
    image = image.gs.write_packing(1e-4, target=BANDS)
    return image.gs.write_rgb(red="B04", green="B03", blue="B02")


def build(raw: Path, out: Path, tiles: int, seed: int, workers: int) -> None:
    picked = candidates(raw, tiles, seed)
    source = StacClient.planetary_computer().source("sentinel-2-l2a")
    source = source.set_config(bands=BANDS)

    def fetch(row):
        label = read_label(row.path)
        return row, label, load_image(source, label)

    done = []
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, row) for row in picked.itertuples()]
        for future in as_completed(futures):
            try:
                row, label, image = future.result()
            except Exception as error:  # a scene Planetary Computer cannot serve
                print(f"skip: {error}")
                continue
            if image.sizes.get("time", 1) != 1:
                print(f"skip {row.dw_id}: {image.sizes['time']} scenes that day")
                continue
            done.append((row, label, image))
            print(f"{len(done)}/{tiles} {row.dw_id}")

    # Biome order, then every sixth tile to val and test, keeps biomes in each split.
    done.sort(key=lambda item: (item[0].biome, item[0].dw_id))
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(len(done), size=min(tiles, len(done)), replace=False))
    write_classes(out)
    for number, index in enumerate(keep):
        row, label, image = done[index]
        split = out / SPLITS[number % len(SPLITS)]
        for kind, raster in (("images", image), ("labels", label)):
            (split / kind).mkdir(parents=True, exist_ok=True)
            raster.gs.to_cog(split / kind / f"{row.dw_id}.tif", **COMPRESSION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="extracted Dynamic World release")
    parser.add_argument("out", type=Path)
    parser.add_argument("--tiles", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    build(
        arguments.raw, arguments.out, arguments.tiles, arguments.seed, arguments.workers
    )
