"""Native model tensor conversion and numerical processing contracts."""

import numpy as np
import pytest
import torch

from geosave.model.processing import normalize, to_raster


def test_normalization_uses_the_declared_axis_and_preserves_gradients():
    image = torch.full((2, 3, 2, 4, 4), 5.0, requires_grad=True)
    result = normalize(image, mean=[1, 2], std=[2, 4], dim=-3)
    assert result.shape == image.shape
    torch.testing.assert_close(result[:, :, 0], torch.full((2, 3, 4, 4), 2.0))
    torch.testing.assert_close(result[:, :, 1], torch.full((2, 3, 4, 4), 0.75))
    result.sum().backward()
    assert (image.grad[:, :, 0] == 0.5).all()
    assert (image.grad[:, :, 1] == 0.25).all()
    assert (image == 5).all()


def test_normalization_preserves_missing_pixels_and_rejects_invalid_statistics():
    image = torch.tensor([[[float("nan"), float("inf"), 2.0]]])
    result = normalize(image, mean=[1], std=[2], dim=0)
    assert result.isnan()[0, 0, 0]
    assert result.isinf()[0, 0, 1]
    assert result[0, 0, 2] == 0.5
    with pytest.raises(ValueError, match="positive"):
        normalize(image, mean=[1], std=[0], dim=0)
    with pytest.raises(ValueError, match="match"):
        normalize(image, mean=[1, 2], std=[1, 2], dim=0)


def test_output_grid_mismatch_is_rejected_instead_of_relabelled():
    from tests.model.spec.test_processing import scene

    with pytest.raises(ValueError, match="reference grid"):
        to_raster(
            np.zeros((1, 4, 4)),
            reference=scene(),
            dims=["band", "y", "x"],
            name="prediction",
        )


def test_tensor_conversion_never_broadcasts_a_static_band_over_time():
    import xarray as xr
    from geosave.model.processing import to_tensor

    raster = xr.Dataset(
        {
            "red": (("time", "y", "x"), np.ones((2, 2, 2))),
            "nir": (("y", "x"), np.ones((2, 2))),
        }
    )
    with pytest.raises(ValueError, match="dimension"):
        to_tensor(raster, dims=["band", "time", "y", "x"])
