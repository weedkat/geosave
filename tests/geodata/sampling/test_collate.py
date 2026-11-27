import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

import geosave as gs


def build_scene(height: int) -> tuple[xr.Dataset, xr.DataArray]:
    image = xr.Dataset({"red": (("y", "x"), np.full((height, 4), height, "float32"))})
    label = xr.DataArray(np.zeros((height, 4), "uint8"), dims=("y", "x"))
    return image, label


def test_collate_keeps_scenes_of_different_sizes_as_lists():
    scenes = [build_scene(3), build_scene(4), build_scene(5)]

    images, labels = next(iter(DataLoader(scenes, batch_size=2, collate_fn=gs.collate)))

    assert isinstance(images, list)
    assert [image.sizes["y"] for image in images] == [3, 4]
    xr.testing.assert_identical(images[1], scenes[1][0])
    xr.testing.assert_identical(labels[0], scenes[0][1])


def test_collate_stacks_tensors_as_default_collate_does():
    sample = (
        torch.zeros(4, 8, 8),
        {"centroid": torch.tensor([1.0, 2.0])},
        torch.zeros(8, 8, dtype=torch.long),
    )

    pixels, context, target = gs.collate([sample, sample])

    assert pixels.shape == (2, 4, 8, 8)
    assert context["centroid"].shape == (2, 2)
    assert target.shape == (2, 8, 8)


def test_collate_keeps_trees_as_lists():
    tree = xr.DataTree.from_dict({"image": build_scene(3)[0]})

    batch = gs.collate([{"window": tree, "index": 0}, {"window": tree, "index": 1}])

    assert batch["window"] == [tree, tree]
    assert batch["index"].tolist() == [0, 1]
