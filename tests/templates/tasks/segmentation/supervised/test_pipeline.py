"""Prediction from a stored observation through the starter's processing stages."""

import numpy as np
import pytest
import torch
import xarray as xr
from odc.geo.geobox import GeoBox
from generated_kits.segmentation.supervised.processing.persistor import persist
from generated_kits.segmentation.supervised.processing.postprocessor import postprocess
from generated_kits.segmentation.supervised.processing.predictor import predict
from generated_kits.segmentation.supervised.processing.preprocessing import Reflectance

from geosave.geodata.io import open_raster, write_cog
from tests.conftest import build_dataset


class TwoBandModel(torch.nn.Module):
    def forward(self, image, **context):
        return torch.stack((image[:, 1], image[:, 0]), dim=1)


def test_segmentation_prediction_roundtrips_on_the_source_grid(tmp_path):
    box = GeoBox.from_bbox((0, 0, 170, 130), crs="EPSG:32633", resolution=10)
    rows, columns = np.indices(box.shape.yx)
    expected = columns > rows
    bands = {
        name: (("y", "x"), np.where(pattern, 4000, 1000).astype("uint16"))
        for name, pattern in zip(
            ["B02", "B03", "B04", "B08"],
            [expected, ~expected, expected, expected],
            strict=True,
        )
    }
    source = build_dataset(bands, box, time=np.datetime64("2020-03-05T10:00:00"))
    for layer in source.data_vars.values():
        layer.attrs.update(scale_factor=0.0001, add_offset=0.0)
    source_path = tmp_path / "imagery.tif"
    write_cog(source, source_path)

    model = TwoBandModel()
    with open_raster(source_path) as opened:
        tree = xr.DataTree.from_dict({"image": opened})
        image = Reflectance().preprocess(tree)["image"].gs.to_dataset()
        with pytest.raises(ValueError, match="model.eval"):
            predict(model, image, size=8)
        logits = predict(model.eval(), image, size=8, batch_size=3)
    product = postprocess(logits)
    path = tmp_path / "prediction.tif"
    persist(product, path)
    with open_raster(path) as result:
        assert result.gs.geobox == box
        assert result.time.values == source.time.values
        assert result.class_id.dtype == np.int32
        assert result.class_id.rio.nodata == -1
        np.testing.assert_array_equal(result.class_id.to_numpy(), expected)
