# ruff: noqa
# Throwaway prototype for docs/contributing/design/catalog-to-training.md; not library code.
"""Importable sample loader: what a worker process needs, nothing run at import."""

from functools import cache
from pathlib import Path

import geosave as gs
from cube import open_cube, select_slots
from plan import read_geobox

COLLECTIONS = ("s2", "s1", "labels")


@cache
def open_tables(root: str):
    catalog = gs.open_vector(Path(root) / "catalog.parquet")
    plan = gs.open_vector(Path(root) / "samples/plan.parquet")
    return catalog, plan


def load_sample(catalog, sample) -> dict:
    """Every source collection of one sample on the sample's grid, one step per slot."""
    grid = read_geobox(sample)
    arrays = {}
    for collection in COLLECTIONS:
        slots = list(sample[collection])
        rows = catalog[catalog["id"].isin([value for value in slots if value])]
        cube = select_slots(open_cube(rows, grid), slots).load()
        arrays[collection] = (
            cube.where(cube["valid"], 0)
            .to_array("band")
            .values.astype(cube.to_array().dtype)
        )
        arrays[f"{collection}_valid"] = cube["valid"].values
    return arrays


class Prepare:
    """LitData `fn`: sample position → record."""

    def __init__(self, root: str, repeat: int):
        self.root = root
        self.repeat = repeat

    def __call__(self, index: int) -> dict:
        catalog, plan = open_tables(self.root)
        record = load_sample(catalog, plan.iloc[index % len(plan)])
        record["index"] = index
        return record
