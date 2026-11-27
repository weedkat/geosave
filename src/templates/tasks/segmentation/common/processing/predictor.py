"""Predict overlapping chips and merge logits on their source grid."""

from collections.abc import Callable
from itertools import batched

import torch
import xarray as xr
from torch import nn
from torch.utils.data import default_collate

import geosave as gs


@torch.inference_mode()
def predict(
    model: nn.Module,
    image: xr.Dataset,
    *,
    size: int,
    overlap: int | float = 0.5,
    batch_size: int = 8,
    context: dict[str, Callable[[xr.Dataset], torch.Tensor] | str] | None = None,
) -> xr.DataArray:
    """Return per-class logits from an evaluation-mode model on the image grid."""
    if model.training:
        raise ValueError("Call model.eval() before scene prediction.")
    readers: dict[str, Callable[[xr.Dataset], torch.Tensor] | str] = {
        "transform": "transform",
        "crs": "crs",
    }
    if context is not None:
        readers.update(context)
    chips = gs.RasterChipper(image, size=size, overlap=overlap)
    merger = gs.RasterMerger(chips, window="hann")
    device = next(model.parameters(), torch.empty(0)).device

    for group in batched(chips, batch_size):
        samples = []
        for _, chip in group:
            pixels, metadata = chip.gs.to_tensor(dtype="float32", context=readers)
            if not torch.isfinite(pixels).all():
                raise ValueError("Handle missing image values in preprocessing.")
            samples.append({"image": pixels, **metadata})
        inputs = default_collate(samples)
        logits = model(**{name: value.to(device) for name, value in inputs.items()})
        for (location, _), scores in zip(
            group, logits.float().cpu().numpy(), strict=True
        ):
            merger.add(location, scores)
    (logits,) = merger.merge()
    if "time" in image.coords and image.time.ndim == 0:
        logits = logits.assign_coords(time=image.time)
    return logits
