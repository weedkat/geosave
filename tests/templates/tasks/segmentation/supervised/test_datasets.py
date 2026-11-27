"""The raster dataset prepares images with the configured preprocessor."""

import numpy as np
from odc.geo.geobox import GeoBox

from geosave.geodata.io import write_cog
from generated_kits.segmentation.supervised.training.datasets import RasterDataset
from tests.conftest import build_dataset

BANDS = ["B02", "B03", "B04", "B08"]


def test_a_raster_pair_reads_as_reflectance_and_class_ids(tmp_path):
    grid = GeoBox.from_bbox((0, 0, 80, 80), crs="EPSG:32633", resolution=10)
    image = build_dataset(
        {band: (("y", "x"), np.full((8, 8), 2000, dtype="uint16")) for band in BANDS},
        grid,
    )
    for variable in image.data_vars.values():
        variable.attrs.update(scale_factor=0.0001, add_offset=0.0)
    class_ids = np.eye(8, dtype="uint8")
    class_ids[0, 0] = 255
    labels = build_dataset({"class_id": (("y", "x"), class_ids)}, grid)
    write_cog(image, tmp_path / "image.tif")
    write_cog(labels, tmp_path / "label.tif")

    dataset = RasterDataset(
        images=[str(tmp_path / "image.tif")],
        targets=[str(tmp_path / "label.tif")],
        preprocessor={
            "class_path": "generated_kits.segmentation.supervised.processing.preprocessing.Reflectance"
        },
        num_classes=2,
        context={"transform": "transform", "crs": "crs", "centroid": "centroid"},
    )
    sample = dataset[0]

    assert set(sample) == {"inputs", "target"}
    inputs = sample["inputs"]
    assert inputs["image"].shape == (4, 8, 8)
    assert np.allclose(inputs["image"].numpy(), 0.2)
    np.testing.assert_array_equal(sample["target"].numpy(), class_ids)
    assert inputs["crs"].item() == 32633
    assert inputs["centroid"].shape == (2,)
