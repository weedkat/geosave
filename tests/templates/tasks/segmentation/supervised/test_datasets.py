"""The template refuses recipe-based reads until ModelSpec is rebuilt."""

import pytest

from modules.segmentation.supervised.datasets import RasterDataset


def test_recipe_dataset_fails_before_opening_pixels():
    with pytest.raises(NotImplementedError, match="ModelSpec is being redesigned"):
        RasterDataset(
            images=["missing-image.tif"],
            targets=["missing-label.tif"],
            spec="missing-recipe.yaml",
            num_classes=2,
        )
