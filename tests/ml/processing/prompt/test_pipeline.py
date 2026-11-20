"""A prompt pipeline carries map-coordinate prompts beside the sample to every step."""

import geopandas as gpd
import numpy as np
import rasterio.features
import shapely
import torch
import xarray as xr

import geosave  # noqa: F401  registers the .gs accessors
from geosave.ml import (
    DensePreprocessor,
    PromptPipeline,
    PromptPostprocessor,
    PromptPredictor,
)
from tests.geodata.sampling.conftest import build_scene


class BoxModel(torch.nn.Module):
    """Stands in for SAM: the mask of a box prompt is the box."""

    def forward(self, image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(image.shape[-2:], dtype=torch.bool)
        left, top, right, bottom = box.round().int().tolist()
        mask[top:bottom, left:right] = True
        return mask


class BoxPrompts(PromptPredictor[gpd.GeoDataFrame]):
    def __call__(
        self, model: torch.nn.Module, sample: xr.DataTree, prompts: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        image = sample["image"].to_dataset()
        grid = image.gs.geobox
        pixels = torch.from_numpy(image.to_dataarray("band").values)
        masks = []
        for box in prompts.to_crs(grid.crs.to_wkt()).geometry:
            left, bottom, right, top = box.bounds
            col0, row0 = ~grid.affine * (left, top)
            col1, row1 = ~grid.affine * (right, bottom)
            with torch.inference_mode():
                mask = model(pixels, torch.tensor([col0, row0, col1, row1])).numpy()
            shapes = rasterio.features.shapes(
                mask.astype("uint8"), mask=mask, transform=grid.affine
            )
            masks.append(
                shapely.union_all([shapely.geometry.shape(s) for s, _ in shapes])
            )
        output = gpd.GeoDataFrame(
            geometry=masks, crs=grid.crs.to_wkt(), index=prompts.index
        )
        return output


class KeepAttributes(PromptPostprocessor[gpd.GeoDataFrame, gpd.GeoDataFrame]):
    def __call__(
        self, output: gpd.GeoDataFrame, sample: xr.DataTree, prompts: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        product = gpd.GeoDataFrame(output.join(prompts.drop(columns="geometry")))
        return product


def test_a_prompt_pipeline_answers_each_prompt_in_map_coordinates():
    scene = build_scene(times=None)
    grid = scene.gs.geobox
    left, bottom, right, top = grid.extent.boundingbox
    size = grid.resolution.x
    prompts = gpd.GeoDataFrame(
        {"field": ["north", "south"]},
        geometry=[
            shapely.box(left, top - 3 * size, left + 2 * size, top),
            shapely.box(left + 4 * size, bottom, left + 8 * size, bottom + 2 * size),
        ],
        crs=grid.crs.to_wkt(),
    )
    pipeline = PromptPipeline(
        BoxModel(),
        DensePreprocessor(["red", "nir"], decode=False),
        BoxPrompts(),
        KeepAttributes(),
    )

    product = pipeline.predict(xr.DataTree.from_dict({"image": scene}), prompts)

    assert product["field"].tolist() == ["north", "south"]
    np.testing.assert_allclose(product.area, prompts.area)
