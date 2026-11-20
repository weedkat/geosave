"""A change pipeline prepares two acquisitions as one sample and reuses the raster predictor."""

import torch
import xarray as xr

import geosave  # noqa: F401  registers the .gs accessors
from geosave.ml import ChangePipeline, ChangePreprocessor, DensePredictor
from geosave.ml.processing.raster.postprocessing.base import RasterPostprocessor
from geosave.ml.processing.raster.postprocessing.dense import copy_scalar_coords
from tests.geodata.sampling.conftest import build_scene


class Stacked(ChangePreprocessor):
    def __call__(self, before: xr.DataTree, after: xr.DataTree) -> xr.DataTree:
        early = before["image"].to_dataset().astype("float32")
        late = after["image"].to_dataset().astype("float32")
        sample = xr.DataTree.from_dict({"before": early, "after": late})
        return sample


class Difference(torch.nn.Module):
    """Stands in for a siamese change network: the mean absolute band difference."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        return self.scale * (after - before).abs().mean(1, keepdim=True)


class Changed(RasterPostprocessor[xr.DataArray, xr.Dataset]):
    def __call__(self, output: xr.DataArray, sample: xr.DataTree) -> xr.Dataset:
        changed = (output.isel(band=0, drop=True) > 1).astype("uint8")
        product = copy_scalar_coords(changed.to_dataset(name="changed"), sample)
        return product


def test_a_change_pipeline_finds_what_changed_between_two_acquisitions():
    scene = build_scene(times=None)
    cleared = scene.copy(deep=True).compute()
    cleared["red"][:3, :4] = cleared["red"][:3, :4] + 100
    pipeline = ChangePipeline(
        Difference().eval(), Stacked(), DensePredictor(size=4, stride=3), Changed()
    )

    product = pipeline.predict(
        xr.DataTree.from_dict({"image": scene}),
        xr.DataTree.from_dict({"image": cleared}),
    )

    assert int(product["changed"].sum()) == 12
    assert product["changed"].values[:3, :4].all()
    assert product.gs.geobox == scene.gs.geobox
