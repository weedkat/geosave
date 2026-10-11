"""Turn a sample's rasters into the named tensors a model takes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import xarray as xr

from geosave_engine.model.spec import ModelSpec


def model_inputs(
    spec: ModelSpec,
    rasters: Mapping[str, xr.Dataset | xr.DataArray],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare identical model inputs for a training or inference sample.

    Args:
        spec: Raster bindings and optional context recipe.
        rasters: Prepared input rasters, as the sample's window read them.
        context: Explicit cached encoding for these rasters and recipe.

    Returns:
        Named tensors retaining raster time/band axes.
    """
    values = spec.model_inputs(rasters, context=context)
    return {
        name: value.gs.to_tensor()
        if isinstance(value, (xr.Dataset, xr.DataArray))
        else torch.as_tensor(value)
        for name, value in values.items()
    }
