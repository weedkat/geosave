"""Predict with this project's trained model."""

import xarray as xr
from torch import nn


def predict(model: nn.Module, inputs: xr.DataTree):
    """Apply the model to prepared inputs using the project's sampling policy."""
    raise NotImplementedError(
        "Implement spatial inference for continuous model outputs for this project."
    )
