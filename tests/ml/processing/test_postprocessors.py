"""The class map decodes merged logits into named classes."""

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.ml.processing.postprocessors import NO_PREDICTION, ClassMap


def test_postprocess_names_classes_and_marks_missing_logits():
    decoder = ClassMap(["water", "trees", "crops"], ["#419bdf", "#397d49", "#e49635"])
    geobox = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    values = np.zeros((3, 2, 2), dtype="float32")
    values[2, 0, 0] = 5.0
    values[:, 1, 1] = np.nan
    logits = xr.DataArray(
        values, dims=("band", "y", "x"), coords=xr_coords(geobox, always_yx=True)
    )

    class_map = decoder.postprocess(logits)

    assert class_map["class_id"].dtype == "uint8"
    assert class_map["class_id"].values.tolist() == [[2, 0], [0, NO_PREDICTION]]
    assert class_map.gs.geobox == geobox
    assert "crops" in str(class_map["class_id"].attrs)
