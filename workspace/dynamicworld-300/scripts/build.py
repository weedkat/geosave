"""Build the ``fatmur/dynamicworld-300`` teaching subset from the Dynamic World data repo.

The subset keeps the data repo's layout: original labels, one shared image and
SCL per scene, both in stored digital numbers, and one ``scene.parquet`` row per
annotation. Scenes are taken in a seeded biome round robin per split, so any
prefix spreads evenly over biomes. DATA is a checkout of ``fatmur/dynamicworld``
after its EDA; run from this repository so ``release`` writes into it:

    python scripts/build.py select DATA STAGING
    python DATA/scripts/ingest_scenes.py STAGING/label.parquet --threads 8
    python scripts/build.py release DATA STAGING .

``select`` stages a few spare scenes per split; ``release`` keeps the first
written scenes of each split, writes their images with the teaching bands and
copies their SCL.
"""

import shutil
from pathlib import Path

import geopandas as gpd
import typer
from tqdm import tqdm

import geosave as gs
from geosave.geodata import io

SCENES = {"train": 200, "val": 50, "test": 50}
SPARE = 10
BANDS = ["B02", "B03", "B04", "B08"]

app = typer.Typer()


def rank_scenes(labels_df: gpd.GeoDataFrame, split: str) -> list[str]:
    """Order a split's prepared scenes so each prefix alternates across biomes."""
    scenes_df = labels_df.loc[
        labels_df["split"].eq(split) & labels_df["label_path"].notna()
    ].drop_duplicates("scene_id")
    scenes_df = scenes_df.sample(frac=1, random_state=0)
    scenes_df["turn"] = scenes_df.groupby("biome").cumcount()
    scenes_df = scenes_df.sort_values("turn", kind="stable")

    return scenes_df["scene_id"].tolist()


@app.command()
def select(data: Path, staging: Path) -> None:
    """Stage ranked candidate scenes, their labels and an ingestion table."""
    processed = data / "processed"
    labels_df = gpd.read_parquet(processed / "label.parquet")
    prepared_df = labels_df.loc[labels_df["label_path"].notna()]

    scene_ids = []
    for split, count in SCENES.items():
        scene_ids += rank_scenes(labels_df, split)[: count + SPARE]

    # Rows follow scene rank, so `release` reads the rank back from row order.
    selected_df = prepared_df.loc[prepared_df["scene_id"].isin(scene_ids)].copy()
    selected_df["rank"] = selected_df["scene_id"].map(
        {scene_id: rank for rank, scene_id in enumerate(scene_ids)}
    )
    selected_df = selected_df.sort_values("rank", kind="stable").drop(columns="rank")

    for label_path in tqdm(selected_df["label_path"], desc="Copy labels"):
        destination = staging / label_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(processed / label_path, destination)

    selected_df.to_parquet(staging / "label.parquet", index=False)
    print(selected_df.drop_duplicates("scene_id")["split"].value_counts().to_string())


@app.command()
def release(data: Path, staging: Path, destination: Path) -> None:
    """Keep the first written scenes of each split with the teaching bands."""
    scenes_df = gpd.read_parquet(staging / "scene.parquet")
    written_df = scenes_df.loc[scenes_df["status"].eq("written")]

    scene_ids = []
    for split, count in SCENES.items():
        split_ids = written_df.loc[written_df["split"].eq(split), "scene_id"].unique()
        if len(split_ids) < count:
            raise typer.BadParameter(f"{split} has {len(split_ids)} written scenes")
        scene_ids += list(split_ids[:count])

    kept_df = written_df.loc[written_df["scene_id"].isin(scene_ids)]

    for label_path in tqdm(kept_df["label_path"], desc="Copy labels"):
        target = destination / label_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / label_path, target)

    images_df = kept_df.drop_duplicates("scene_id")
    for scene in tqdm(
        images_df.itertuples(), total=len(images_df), desc="Write images"
    ):
        target = destination / scene.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with gs.open_raster(staging / scene.image_path) as image:
            # Horizontal differencing roughly halves DEFLATE size for reflectance.
            io.write_cog(
                image[BANDS], target, overwrite=True, compress="DEFLATE", predictor=2
            )

        scl_target = destination / scene.scl_path
        scl_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / scene.scl_path, scl_target)

    kept_df.to_parquet(destination / "scene.parquet", index=False)
    shutil.copy2(data / "labels.yaml", destination / "labels.yaml")
    print(kept_df.groupby(["split", "annotation_source"]).size().to_string())


if __name__ == "__main__":
    app()
