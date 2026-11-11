"""Tile layout over xarray sources, with native Tiler merging."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from operator import index as integer_index
from typing import Literal, cast

import numpy as np
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from tiler import Merger, Tiler

from .transform.extent import Padding

type SampleIndex = int
type TileId = int
type ChipKey = tuple[SampleIndex, TileId]
type Mode = Padding | Literal["drop"]
type Window = Literal[
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
    "overlap-tile",
]


class Chips:
    """Lay fixed-size tiles over georeferenced sources and merge predictions back.

    Args:
        samples: Datasets or DataArrays, each on one regular grid.
        size: Tile height and width in pixels.
        overlap: Tiler overlap in pixels or as a fraction of tile size.
        mode: How tiles past a source edge are filled, as in `numpy.pad`.
            ``drop`` omits incomplete tiles instead.
        constant_value: Fill value for ``constant`` mode.

    Indexing returns ``samples[i].gs.crop_to_grid(chips.geobox(key))``: a lazy xarray
    chip whose coordinates, grid and extent describe the chip, keeping time and
    other coordinates for model context. Construction reads only metadata.
    Converting chips to arrays or tensors is the caller's step, such as
    ``chip.gs.to_numpy()`` in a torch Dataset.

    Examples:
        >>> chips = Chips([scene], size=224, overlap=32, mode="reflect")
        >>> chip = chips[3]
        >>> chip.gs.region.extent.centroid
        <Geometry ...>
        >>> outputs = chips.merge((key, predict(chips[key])) for key in chips)
    """

    def __init__(
        self,
        samples: Sequence[xr.Dataset | xr.DataArray],
        size: int,
        overlap: int | float = 0,
        *,
        mode: Mode = "constant",
        constant_value: int | float = 0,
    ) -> None:
        if mode not in ("constant", "reflect", "edge", "wrap", "drop"):
            raise ValueError(
                "Chips requires a fixed-size mode: constant, reflect, edge, wrap or drop"
            )
        self.xarray = tuple(samples)
        self._mode = mode
        self._constant_value = constant_value
        self._tilers: list[Tiler] = []
        self._offsets = [0]
        for sample in self.xarray:
            sample.gs.validate(affine=True)
            grid = cast("GeoBox", sample.gs.geobox)
            tiler = Tiler(
                data_shape=tuple(grid.shape),
                tile_shape=(size, size),
                overlap=overlap,
                mode=mode,
                constant_value=constant_value,
            )
            self._tilers.append(tiler)
            self._offsets.append(self._offsets[-1] + len(tiler))

    def __len__(self) -> int:
        """Return the total number of tiles across all samples."""
        return self._offsets[-1]

    def key(self, index: int) -> ChipKey:
        """Return the sample and tile ID for a flat, nonnegative chip index."""
        index = integer_index(index)
        if index < 0 or index >= len(self):
            raise IndexError("chip index out of range")
        sample_index = bisect_right(self._offsets, index) - 1
        return sample_index, index - self._offsets[sample_index]

    def geobox(self, key: int | ChipKey) -> GeoBox:
        """Return a chip's pixel grid, including padded edges, without reading pixels.

        Args:
            key: Flat index or (sample_index, tile_id), as accepted by indexing.

        Returns:
            Grid on the source CRS and pixel size, covering the full chip.

        Raises:
            IndexError: The chip index is outside the inventory.
        """
        sample_index, tile_id = key if isinstance(key, tuple) else self.key(key)
        if sample_index < 0:
            raise IndexError("sample_index must be nonnegative")
        grid = self.xarray[sample_index].gs.geobox
        (top, left), (bottom, right) = self._tilers[sample_index].get_tile_bbox(tile_id)
        transform = grid.transform * Affine.translation(int(left), int(top))
        return GeoBox((int(bottom - top), int(right - left)), transform, grid.crs)

    def __getitem__(self, key: int | ChipKey) -> xr.Dataset | xr.DataArray:
        """Return one lazy chip by flat index or explicit (sample_index, tile_id)."""
        sample_index, _ = key if isinstance(key, tuple) else self.key(key)
        mode = "constant" if self._mode == "drop" else self._mode
        return self.xarray[sample_index].gs.crop_to_grid(
            self.geobox(key), mode=mode, constant_value=self._constant_value
        )

    def __iter__(self) -> Iterator[ChipKey]:
        """Yield every chip key in sample order."""
        for sample_index, tiler in enumerate(self._tilers):
            for tile_id in range(len(tiler)):
                yield sample_index, tile_id

    def merge(
        self,
        predictions: Iterable[tuple[ChipKey, np.ndarray]],
        *,
        window: Window | np.ndarray | None = None,
    ) -> list[np.ndarray | None]:
        """Merge predictions with native Tiler weighting in sample order.

        Args:
            predictions: ``((sample_index, tile_id), pixels)`` pairs in any order.
                Spatial axes trail. Leading shapes must agree within a sample.
                Repeated tiles accumulate; missing tiles are allowed.
            window: Forwarded to Merger. Array windows must match its tile shape,
                with leading prediction axes flattened into one channel axis.

        Returns:
            One float32 NumPy array per sample, or None for a sample without
            predictions. Merger normalization fills uncovered pixels with zero.
            Place an output with ``xr.DataArray(...).gs.write_grid(sample.gs.geobox)``.
            Merging allocates full outputs in memory; it is not a streaming writer.

        Raises:
            ValueError: Leading shapes differ within a sample.
        """
        mergers: dict[int, Merger] = {}
        shapes: dict[int, tuple[int, ...]] = {}
        for (sample_index, tile_id), pixels in predictions:
            if sample_index < 0:
                raise IndexError("sample_index must be nonnegative")
            shape = pixels.shape[:-2]
            if sample_index in shapes and shape != shapes[sample_index]:
                raise ValueError("prediction leading shape must agree within a sample")
            values = pixels.reshape((-1, *pixels.shape[-2:]))
            if sample_index not in mergers:
                layout = self._tilers[sample_index]
                tiler = Tiler(
                    data_shape=(values.shape[0], *layout.data_shape),
                    tile_shape=(values.shape[0], *layout.tile_shape),
                    channel_dimension=0,
                    overlap=layout.overlap,
                    mode=layout.mode,
                )
                mergers[sample_index] = Merger(tiler, window=window)
                shapes[sample_index] = shape
            mergers[sample_index].add(tile_id, values)

        outputs: list[np.ndarray | None] = []
        for index in range(len(self._tilers)):
            if index not in mergers:
                outputs.append(None)
                continue
            pixels = mergers[index].merge()
            outputs.append(pixels.reshape((*shapes[index], *pixels.shape[-2:])))
        return outputs
