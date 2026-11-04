"""Rewrite the Dynamic World example tiles with a catalog of every split.

Each image/label pair is written back to the same layout,
``<split>/images/<id>.tif`` and ``<split>/labels/<id>.tif``, and described in
one ``catalog.parquet`` with a ``split`` column, ready to upload as
``fatmur/dynamicworld-300``.

Usage:
    uv run python scripts/build_dynamic_world.py SOURCE DESTINATION
"""

import shutil
import sys
from pathlib import Path

import geosave as gs

SPLITS = ("train", "val", "test")


def build_split(source: Path, destination: Path, split: str):
    """Write one split's pairs and return their catalog."""
    images = sorted((source / split / "images").glob("*.tif"))
    catalogs = []
    for image_path in images:
        label_path = source / split / "labels" / image_path.name
        pair = gs.create_datatree(
            {
                "sentinel-2-l2a": gs.open_raster(image_path),
                "label": gs.open_raster(label_path),
            }
        )
        stem = image_path.stem
        catalogs.append(
            pair.gs.to_catalog(
                destination / split,
                template={
                    "sentinel-2-l2a": f"images/{stem}.tif",
                    "label": f"labels/{stem}.tif",
                },
                id_format=stem,
                # The published tiles use DEFLATE with horizontal differencing.
                compress="DEFLATE",
                predictor=2,
            )
        )
    catalog = gs.GeoCatalog.concat(catalogs)
    catalog["split"] = split
    return catalog


def main(source: Path, destination: Path) -> None:
    catalogs = []
    for split in SPLITS:
        catalogs.append(build_split(source, destination, split))
    catalog = gs.GeoCatalog.concat(catalogs)
    catalog.gs.to_geoparquet(destination / "catalog.parquet")
    for name in ("classes.yaml", "README.md"):
        shutil.copy(source / name, destination / name)
    print(catalog.groupby(["split", "collection"]).size())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
