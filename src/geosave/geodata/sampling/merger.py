"""Merge chip predictions back onto the rasters the chips were cut from."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr
from odc.geo.xr import wrap_xr

from geosave.geodata.core.conventions import BAND_DIMENSION
from geosave.geodata.io import storage

from .layout import ChipLocation, ChipSource

if TYPE_CHECKING:
    from geosave.geodata.io.storage import Location

    from .chipper import RasterChipper

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


@cache
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
    weights = np.maximum(np.outer(taper, taper), _WEIGHT_FLOOR).astype("float32")
    # Cached and shared between calls, so no caller may change it.
    weights.flags.writeable = False
    return weights


class RasterMerger:
    """Average overlapping dense chip predictions back onto the chipper's rasters.

    Built from the chipper that cut the chips, as Tiler's `Merger(tiler)` is,
    so it knows every raster's grid and the chip size before the first chip
    arrives. Each chip predicts a `(band, size, size)` array on its own
    pixels; chips of any raster may arrive in any order. The padded part of a
    chip reaching past its raster is dropped.

    Without a store, each raster's weighted sums are held in memory as
    float32. With a store, a folder, raster `i` accumulates in
    `<store>/<i>.zarr`, so rasters larger than memory merge: each chip adds
    into the chunks it covers.

    Args:
        chips: The chipper the chips come from.
        window: Taper weighting each chip; `"boxcar"` averages overlaps evenly.
        store: Folder to accumulate in; None holds the sums in memory.

    Examples:
        >>> chips = gs.RasterChipper(scenes, size=256, overlap=0.25)
        >>> merger = gs.RasterMerger(chips, window="hann")
        >>> for location, chip in chips:
        ...     merger.add(location, predict(chip))
        >>> scores = merger.merge()
        >>> len(scores) == len(scenes)
        True
    """

    def __init__(
        self,
        chips: RasterChipper[Any],
        *,
        window: MergeWindow = "boxcar",
        store: Location | None = None,
    ) -> None:
        self.sources = chips.sources
        self.size = chips.size
        self.window: MergeWindow = window
        self.store = None if store is None else storage.create_upath(store)
        # Created when a raster's first chip arrives: the bands come from its output.
        self.sums: list[Any] = [None] * len(self.sources)
        self.weights: list[Any] = [None] * len(self.sources)

    def add(self, location: ChipLocation, prediction: np.ndarray) -> None:
        """Add one chip's prediction, weighted by the window.

        Args:
            location: The location yielded with the chip.
            prediction: Values shaped `(band, size, size)`.

        Raises:
            ValueError: The prediction is not `(band, size, size)`, or the
                chip was cut by another chipper.
            FileExistsError: The raster's store already exists.
        """
        if prediction.ndim != 3 or prediction.shape[1:] != (self.size, self.size):
            raise ValueError(
                f"RasterMerger needs dense chip outputs shaped "
                f"(band, {self.size}, {self.size}); got {prediction.shape}"
            )
        source = location.source
        index = source.index
        if index >= len(self.sources) or self.sources[index] != source:
            raise ValueError(
                f"the chip names raster {index}, which this merger's chipper did not cut"
            )
        if self.sums[index] is None:
            self.sums[index], self.weights[index] = self.create_sums(
                source, prediction.shape[0]
            )
        sums = self.sums[index]
        weights = self.weights[index]

        # Padding reaches past the bottom and right edges of the raster.
        height, width = source.geobox.shape
        rows = slice(location.top, min(location.top + self.size, height))
        columns = slice(location.left, min(location.left + self.size, width))
        taper = create_chip_weights(self.window, self.size)[
            : rows.stop - rows.start, : columns.stop - columns.start
        ]
        values = prediction[:, : taper.shape[0], : taper.shape[1]] * taper

        sums[:, rows, columns] = sums[:, rows, columns] + values
        weights[rows, columns] = weights[rows, columns] + taper

    def create_sums(self, source: ChipSource, bands: int) -> tuple[Any, Any]:
        """Return one raster's zeroed sums `(band, y, x)` and weights `(y, x)`.

        Raises:
            FileExistsError: The raster's store already exists.
        """
        shape = tuple(source.geobox.shape)
        if self.store is None:
            sums = np.zeros((bands, *shape), dtype="float32")
            weights = np.zeros(shape, dtype="float32")
            return sums, weights

        import zarr

        path = self.store / f"{source.index}.zarr"
        storage.refuse_existing(path, overwrite=False)
        group = zarr.open_group(
            str(path),
            mode="w-",
            zarr_format=3,
            # zarr refuses empty storage_options for a local path.
            storage_options=dict(path.storage_options) or None,
        )
        # Each chip rewrites the chunks it covers; compressing them costs more
        # time than the staging store saves.
        sums = group.create_array(
            "sums",
            shape=(bands, *shape),
            chunks=(bands, self.size, self.size),
            dtype="float32",
            fill_value=0,
            compressors=None,
        )
        weights = group.create_array(
            "weights",
            shape=shape,
            chunks=(self.size, self.size),
            dtype="float32",
            fill_value=0,
            compressors=None,
        )
        return sums, weights

    def merge(self) -> tuple[xr.DataArray, ...]:
        """Return each raster's weighted mean of its chips, once all are added.

        In memory, the mean is divided into the sums, so the merger holds no
        second raster-sized array and cannot be merged again.

        Returns:
            One float32 DataArray `(band, y, x)` per input raster, in input
            order, on that raster's grid and with its extent; NaN where no
            chip reached. Lazy over the store when one is given.

        Raises:
            ValueError: No chip was added.
        """
        bands = None
        for sums in self.sums:
            if sums is not None:
                bands = sums.shape[0]
                break
        if bands is None:
            raise ValueError("no chip was added to merge")

        merged = []
        for index, source in enumerate(self.sources):
            # A raster too small for one dropped chip has none; it stays NaN.
            if self.sums[index] is None:
                self.sums[index], self.weights[index] = self.create_sums(source, bands)
            mean = self.divide(self.sums[index], self.weights[index])
            dims = (BAND_DIMENSION, *source.geobox.dimensions)
            raster = wrap_xr(mean, source.geobox, axis=1, dims=dims)
            merged.append(raster.gs.write_extent(source.extent))
        return tuple(merged)

    def divide(self, sums: Any, weights: Any) -> Any:
        """Return the sums divided by their weights, NaN where the weight is zero."""
        if self.store is not None:
            import dask.array

            lazy_sums = dask.array.from_zarr(sums)
            lazy_weights = dask.array.from_zarr(weights)
            # Dividing by NaN where no chip reached leaves NaN without a warning.
            return lazy_sums / dask.array.where(lazy_weights > 0, lazy_weights, np.nan)

        covered = weights > 0
        np.divide(sums, weights, out=sums, where=covered)
        sums[:, ~covered] = np.nan
        return sums
