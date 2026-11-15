"""The raster dataset prepares images with the configured preprocessor."""

import numpy as np
from odc.geo.geobox import GeoBox

from geosave.geodata.io import write_cog
from modules.segmentation.supervised.datasets import RasterDataset
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
    labels = build_dataset({"class_id": (("y", "x"), np.eye(8, dtype="uint8"))}, grid)
    write_cog(image, tmp_path / "image.tif")
    write_cog(labels, tmp_path / "label.tif")

    dataset = RasterDataset(
        images=[str(tmp_path / "image.tif")],
        targets=[str(tmp_path / "label.tif")],
        preprocessor={
            "class_path": "modules.segmentation.supervised.processing.Reflectance"
        },
        num_classes=2,
    )
    sample = dataset[0]

    assert sample["inputs"]["image"].shape == (4, 8, 8)
    assert np.allclose(sample["inputs"]["image"].numpy(), 0.2)
    assert sample["target"].tolist() == np.eye(8, dtype="int64").tolist()
