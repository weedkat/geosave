"""Merge chip predictions back onto the rasters their layout was laid over."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from odc.geo.xr import xr_coords

from geosave.geodata.core.conventions import BAND_DIMENSION

from .layout import ChipLayout

# Tapers that weight a chip's centre over its rim, as scipy names them.
type MergeWindow = Literal[
    "boxcar",
    "triang",
    "blackman",
    "hamming",
    "hann",
    "bartlett",
    "parzen",
    "bohman",
    "blackmanharris",
    "nuttall",
    "barthann",
]

# A taper reaches zero at a chip's rim; a pixel only rims cover must keep its
# value, so no weight falls below this.
_WEIGHT_FLOOR = 1e-3


def create_chip_weights(window: MergeWindow, size: int) -> np.ndarray:
    """Weight each pixel of a chip by a 2D taper, never down to zero.

    Args:
        window: Window name, as `scipy.signal.get_window` takes it.
            `"boxcar"` weights every pixel alike.
        size: Chip height and width in pixels.

    Returns:
        float32 weights shaped `(size, size)`, largest at the centre.

    Raises:
        ValueError: scipy does not know the window.

    Examples:
        >>> create_chip_weights("hann", 4).round(3)[1]
        array([0.001, 0.562, 0.562, 0.001])
    """
    # Imported here so that importing geodata does not load scipy.signal.
    from scipy.signal import get_window

    taper = get_window(window, size, fftbins=False)
    weights = np.maximum(np.outer(taper, taper), _WEIGHT_FLOOR)
    return weights.astype("float32")


class RasterMerger:
    """Accumulate dense chip predictions and merge them per raster.

    The merger holds the layout, which knows each chip's raster and place,
    so a prediction needs only its chip index and may arrive in any order.
    Overlapping chips are averaged with the window's weights, and the padded
    part of a chip that reaches past its raster is discarded. Merge logits or
    probabilities, then decode.

    Each raster's weighted sums are held in memory as float32 until `merge`,
    which allocates the result beside them. `is_complete` says when a raster
    has all its chips, and `discard` frees it once merged, so a long run
    holds only the rasters still being predicted.

    Args:
        layout: The layout the predicted chips were cut by.
        window: Taper weighting each chip; `"boxcar"` averages overlaps evenly.

    Examples:
        >>> chipper = RasterChipper([tree], size=256, stride=224)
        >>> merger = RasterMerger(chipper.layout, window="hann")
        >>> for index in range(len(chipper)):
        ...     merger.add(index, predict(chipper[index]))
        >>> classes = merger.merge(0).argmax("band")
    """

    def __init__(self, layout: ChipLayout, *, window: MergeWindow = "boxcar") -> None:
        self.layout = layout
        self._chip_weights = create_chip_weights(window, layout.size)
        raster_count = len(layout.geoboxes)
        self._sums: list[np.ndarray | None] = [None] * raster_count
        self._weights: list[np.ndarray | None] = [None] * raster_count
        self._added = [0] * raster_count

    def add(self, index: int, prediction: np.ndarray) -> None:
        """Accumulate one chip's prediction onto its raster.

        Args:
            index: The chip's index in the layout.
            prediction: Values shaped `(band, size, size)`.

        Raises:
            IndexError: The chip index is outside the chips.
            ValueError: The prediction is not `(band, size, size)`, or its
                band count differs from the raster's earlier chips.
        """
        size = self.layout.size
        if prediction.ndim != 3 or prediction.shape[1:] != (size, size):
            raise ValueError(
                f"prediction must be shaped (band, {size}, {size}), "
                f"got {prediction.shape}"
            )

        raster_index = self.layout.locate(index).raster_index
        scene = self.layout.geoboxes[raster_index]
        chip = self.layout.geobox(index)

        sums = self._sums[raster_index]
        weights = self._weights[raster_index]
        if sums is None or weights is None:
            sums = np.zeros((prediction.shape[0], *scene.shape), dtype="float32")
            weights = np.zeros(tuple(scene.shape), dtype="float32")
            self._sums[raster_index] = sums
            self._weights[raster_index] = weights
        if prediction.shape[0] != sums.shape[0]:
            raise ValueError(
                f"prediction has {prediction.shape[0]} bands; this raster's "
                f"earlier chips had {sums.shape[0]}"
            )

        # The part of the chip on the raster, in the raster's pixels and the chip's.
        scene_rows, scene_columns = scene.overlap_roi(chip)
        chip_rows, chip_columns = chip.overlap_roi(scene)
        weighted = prediction * self._chip_weights
        sums[:, scene_rows, scene_columns] += weighted[:, chip_rows, chip_columns]
        weights[scene_rows, scene_columns] += self._chip_weights[chip_rows, chip_columns]
        self._added[raster_index] += 1

    def is_complete(self, raster_index: int) -> bool:
        """Say whether every chip of a raster was added, each once.

        Raises:
            IndexError: The layout has no such raster.
        """
        return self._added[raster_index] == self.layout.count_chips(raster_index)

    def discard(self, raster_index: int) -> None:
        """Free a raster's accumulated predictions, as after merging it.

        Raises:
            IndexError: The layout has no such raster.
        """
        self._sums[raster_index] = None
        self._weights[raster_index] = None
        self._added[raster_index] = 0

    def merge(self, raster_index: int = 0) -> xr.DataArray:
        """Return one raster's merged predictions on its own grid.

        Args:
            raster_index: Position of the raster in the layout's `geoboxes`.

        Returns:
            float32 DataArray shaped `(band, y, x)` with the raster's
            coordinates; NaN where no chip landed.

        Raises:
            IndexError: The layout has no such raster.
            ValueError: No prediction was added for the raster.
        """
        sums = self._sums[raster_index]
        weights = self._weights[raster_index]
        if sums is None or weights is None:
            raise ValueError(f"no prediction was added for raster {raster_index}")

        covered = weights > 0
        merged = np.full(sums.shape, np.nan, dtype="float32")
        # Dividing into `merged` in place avoids whole-raster temporaries.
        np.divide(sums, weights, out=merged, where=covered)
        scene = self.layout.geoboxes[raster_index]
        coords = xr_coords(scene, always_yx=True)
        return xr.DataArray(merged, dims=(BAND_DIMENSION, "y", "x"), coords=coords)
