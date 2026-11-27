import numpy as np
import xarray as xr
from jsonargparse import ArgumentParser
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import xr_coords
from shapely import Polygon

from geosave.geodata.sampling import RasterChipper
from geosave.ml.pipeline import PredictOutput
from geosave.ml.pipeline.postprocessing import Postprocessor, SegmentationPostprocessor

GRID = GeoBox.from_bbox(
    (500_000.0, 9_000_000.0, 500_060.0, 9_000_060.0), crs="EPSG:32749", shape=(6, 6)
)


def build_window(extent: Geometry | None = None) -> xr.DataTree:
    """One timeless model input on GRID."""
    coords = dict(xr_coords(GRID, always_yx=True))
    image = xr.Dataset({"red": (("y", "x"), np.zeros(GRID.shape))}, coords=coords)
    if extent is not None:
        image = image.gs.write_extent(extent)
    return xr.DataTree.from_dict({"image": image})


def chip_scores(size: int = 4, overlap: int = 2, classes: int = 3) -> PredictOutput:
    """Each chip scores class 2 highest; chip 0 is left out, as if the model skipped it."""
    chips = RasterChipper(build_window(), size=size, overlap=overlap)

    def score():
        for location, _ in list(chips)[1:]:
            scores = np.zeros((classes, size, size), dtype="float32")
            scores[2] = 1.0
            yield location, scores

    return PredictOutput(chips, score())


def test_each_pixel_keeps_its_highest_scoring_class():
    product = SegmentationPostprocessor(window="hann")(chip_scores()).product

    classes = product.class_id
    assert classes.dtype == "int32"
    assert classes.gs.nodata == -1
    assert product.gs.geobox == GRID
    # Only the missing chip 0 covers the top-left 2 x 2 pixels.
    assert (classes.values[:2, :2] == -1).all()
    assert (classes.values[2:, :] == 2).all()


def test_a_staged_window_merges_like_one_in_memory(tmp_path):
    staged = SegmentationPostprocessor(window="hann")(chip_scores(), tmp_path)

    in_memory = SegmentationPostprocessor(window="hann")(chip_scores())
    assert [path.name for path in tmp_path.iterdir()] == ["0.zarr"]
    np.testing.assert_array_equal(
        staged.product.class_id.values, in_memory.product.class_id.values
    )


def test_the_map_declares_its_window_extent():
    triangle = Geometry(Polygon(GRID.extent.exterior.coords[:3]), GRID.crs)
    chips = RasterChipper(build_window(triangle), size=6)
    outputs = []
    for location, _ in chips:
        outputs.append((location, np.ones((2, 6, 6), dtype="float32")))

    product = SegmentationPostprocessor()(PredictOutput(chips, iter(outputs))).product

    assert product.class_id.gs.extent == triangle


def test_a_config_builds_it_as_a_postprocessor():
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--postprocessor", type=Postprocessor)
    config = parser.parse_object(
        {
            "postprocessor": {
                "class_path": "geosave.ml.pipeline.postprocessing.SegmentationPostprocessor",
                "init_args": {"window": "hann", "fill": 255, "name": "landcover"},
            }
        }
    )

    postprocessor = parser.instantiate_classes(config).postprocessor

    product = postprocessor(chip_scores()).product
    assert product.landcover.values[0, 0] == 255
