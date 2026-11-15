"""Read where and when samples were observed, from the grid entries they carry.

A batch carries `transform`, `crs` and `time` as `DataTree.gs.to_tensor`
writes them. An encoder pretrained on location or date inputs derives them
here, inside its own step, so every model receives the same batch.
"""

from __future__ import annotations

import pandas as pd
import torch
from affine import Affine
from odc.geo import geom


def locate_centres(
    transform: torch.Tensor, crs: list[str], shape: tuple[int, int]
) -> torch.Tensor:
    """Return the centre of each sample's grid as latitude and longitude.

    Args:
        transform: float64 affine coefficients shaped `(batch, 6)`.
        crs: Each sample's CRS, as `"EPSG:<code>"` or WKT.
        shape: The grid's `(y, x)` size in pixels.

    Returns:
        float32 `(batch, 2)` latitude and longitude in degrees, on the
        device of `transform`.

    Examples:
        >>> locate_centres(batch["transform"], batch["crs"], image.shape[-2:])
        tensor([[51.0000, 11.0000]])
    """
    height, width = shape
    coordinates = []
    for coefficients, code in zip(transform.tolist(), crs, strict=True):
        x, y = Affine(*coefficients) * (width / 2, height / 2)
        centre = geom.point(x, y, code).to_crs("EPSG:4326")
        longitude, latitude = centre.coords[0]
        coordinates.append([latitude, longitude])
    return torch.tensor(coordinates, dtype=torch.float32, device=transform.device)


def read_dates(time: torch.Tensor) -> list[pd.DatetimeIndex]:
    """Return each sample's time labels.

    Args:
        time: int64 nanoseconds since 1970 shaped `(batch, time)`.

    Returns:
        One DatetimeIndex per sample, in frame order.

    Raises:
        ValueError: A sample's time label is missing.
    """
    dates = []
    for nanoseconds in time.cpu().numpy():
        labels = pd.to_datetime(nanoseconds)
        if labels.hasnans:
            raise ValueError("every frame needs its time label")
        dates.append(labels)
    return dates
