"""Reflectance decodes stored bands and standardizes batches."""

import numpy as np
import pytest
import torch
import xarray as xr

from geosave.ml.processing.preprocessors import Reflectance


def test_preprocess_decodes_selected_bands_and_fills_missing_pixels():
    stored = xr.DataArray(
        np.array([[1000, 0]], dtype="uint16"),
        dims=("y", "x"),
        attrs={"scale_factor": 1e-4, "add_offset": 0.0, "_FillValue": 0},
    )
    raster = xr.Dataset({"B04": stored, "B08": stored * 2})

    image = Reflectance(bands=["B08"]).preprocess(raster)

    assert list(image.data_vars) == ["B08"]
    assert image["B08"].dtype == "float32"
    np.testing.assert_allclose(image["B08"].values, [[0.2, 0.0]])


def test_normalize_standardizes_each_band():
    image = torch.ones(2, 2, 1, 1)

    normalized = Reflectance(["B04", "B08"], mean=[1.0, 0.0], std=[1.0, 2.0]).normalize(
        image
    )

    assert normalized[:, :, 0, 0].tolist() == [[0.0, 0.5], [0.0, 0.5]]


def test_mean_without_std_is_refused():
    with pytest.raises(ValueError, match="both mean and std"):
        Reflectance(["B04"], mean=[0.1])
