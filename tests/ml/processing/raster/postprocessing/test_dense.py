"""Dense postprocessors decode merged output into a class map or named values."""

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave.ml.processing.raster.postprocessing.dense import (
    NO_PREDICTION,
    RegressionPostprocessor,
    SegmentationPostprocessor,
)


def as_sample(output: xr.DataArray) -> xr.DataTree:
    """Build the sample an output was predicted over, dated and with a mask."""
    image = output.isel(band=0, drop=True).to_dataset(name="red")
    image = image.assign_coords(time=np.datetime64("2025-06-03", "ns"))
    keep = xr.ones_like(image["red"], dtype=bool)
    keep[0, 1] = False
    return xr.DataTree.from_dict(
        {"image": image, "clear": keep.to_dataset(name="clear")}
    )


def test_postprocess_names_classes_and_marks_missing_logits():
    decoder = SegmentationPostprocessor(
        ["water", "trees", "crops"], ["#419bdf", "#397d49", "#e49635"]
    )
    geobox = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    values = np.zeros((3, 2, 2), dtype="float32")
    values[2, 0, 0] = 5.0
    values[:, 1, 1] = np.nan
    logits = xr.DataArray(
        values, dims=("band", "y", "x"), coords=xr_coords(geobox, always_yx=True)
    )

    class_map = decoder(logits, as_sample(logits))

    assert class_map["class_id"].dtype == "uint8"
    assert class_map["class_id"].values.tolist() == [[2, 0], [0, NO_PREDICTION]]
    assert class_map.gs.geobox == geobox
    assert "crops" in str(class_map["class_id"].attrs)


def _values(band_count: int) -> xr.DataArray:
    geobox = GeoBox.from_bbox((0, 0, 20, 20), crs="EPSG:32749", resolution=10)
    values = np.full((band_count, 2, 2), 3.5, dtype="float32")
    values[:, 1, 1] = np.nan
    return xr.DataArray(
        values, dims=("band", "y", "x"), coords=xr_coords(geobox, always_yx=True)
    )


def test_regression_names_each_band_and_keeps_missing_values_as_nodata():
    output = _values(1)

    product = RegressionPostprocessor(["agbd"], units="Mg/ha")(
        output, as_sample(output)
    )

    assert product["agbd"].dtype == "float32"
    assert product["agbd"].attrs["units"] == "Mg/ha"
    assert np.isnan(product["agbd"].values).tolist() == [[False, False], [False, True]]
    assert np.isnan(product["agbd"].gs.nodata)
    assert product.gs.geobox == output.odc.geobox


def test_regression_refuses_another_band_count_than_its_names():
    with pytest.raises(ValueError, match="2 bands for 1 names"):
        RegressionPostprocessor(["agbd"])(_values(2), as_sample(_values(2)))


def test_a_product_takes_its_samples_time_and_mask():
    output = _values(2)
    sample = as_sample(output)

    classes = SegmentationPostprocessor(
        ["a", "b"], ["#000000", "#111111"], mask="clear"
    )(output, sample)
    values = RegressionPostprocessor(["u", "v"], mask="clear")(output, sample)

    assert classes["time"].values == sample["image"]["time"].values
    assert classes["class_id"].dtype == "uint8"
    assert classes["class_id"].values.tolist() == [
        [0, NO_PREDICTION],
        [0, NO_PREDICTION],
    ]
    assert "a" in str(classes["class_id"].attrs)
    assert np.isnan(values["u"].values).tolist() == [[False, True], [False, True]]


def test_a_mask_node_of_several_variables_is_refused():
    output = _values(1)
    sample = as_sample(output)
    crowded = sample["clear"].to_dataset().assign(other=sample["clear"]["clear"])
    sample = xr.DataTree.from_dict(
        {"image": sample["image"].to_dataset(), "clear": crowded}
    )

    with pytest.raises(ValueError, match="must hold one variable"):
        RegressionPostprocessor(["agbd"], mask="clear")(output, sample)
