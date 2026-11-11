"""Native NumPy tiling and merging with indexed xarray sources."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike
from odc.geo.geobox import GeoBox
from tiler import Merger, Tiler

type SampleIndex = int
type TileId = int
type ChipKey = tuple[SampleIndex, TileId]
type Mode = Literal["constant", "reflect", "edge", "wrap", "drop"]
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
    """Convert prepared samples to NumPy, then cut them with Tiler.

    Args:
        samples: Prepared Datasets or DataArrays, each on one regular spatial grid.
        size: Tile height and width in pixels.
        overlap: Tiler overlap in pixels or as a fraction of tile size.
        mode: Fixed-size Tiler boundary mode; no extra source frame or halo is
            added. ``drop`` omits incomplete tiles.
        constant_value: Forwarded to Tiler for constant padding.
        dtype: Forwarded to ``sample.gs.to_numpy(dtype=dtype)``. None preserves
            dtype and requires Dataset variables to share a dtype.

    Construction computes all samples into memory once. Tiles are NumPy arrays
    in ``gs.to_numpy`` order, with spatial axes last. ``xarray[i]`` retains the
    original prepared source for merged array ``i``. Callers own source-file
    lifetime and construct output xarray objects with appropriate coordinates.
    Source objects are retained by reference; do not change their grids after
    construction.

    Examples:
        >>> chips = Chips(samples, size=256, mode="reflect", dtype="float32")
        >>> pixels = chips[0, 2]
        >>> outputs = chips.merge((key, predict(pixels)) for key, pixels in chips)
    """

    def __init__(
        self,
        samples: Sequence[xr.Dataset | xr.DataArray],
        size: int,
        overlap: int | float = 0,
        *,
        mode: Mode = "constant",
        constant_value: int | float = 0,
        dtype: DTypeLike | None = None,
    ) -> None:
        if mode not in ("constant", "reflect", "edge", "wrap", "drop"):
            raise ValueError(
                "Chips requires a fixed-size mode: constant, reflect, edge, wrap or drop"
            )
        self._arrays: list[np.ndarray] = []
        self.xarray = tuple(samples)
        self._tilers: list[Tiler] = []

        for sample in self.xarray:
            self._validate_sample(sample)

            # Flattening leading axes must remain a view on every tile read.
            pixels = np.ascontiguousarray(sample.gs.to_numpy(dtype=dtype))

            array = pixels.reshape((-1, *pixels.shape[-2:]))
            tiler = Tiler(
                data_shape=array.shape,
                tile_shape=(array.shape[0], size, size),
                channel_dimension=0,
                overlap=overlap,
                mode=mode,
                constant_value=constant_value,
            )

            self._arrays.append(pixels)
            self._tilers.append(tiler)

    def __len__(self) -> int:
        """Return the total number of tiles across all samples."""
        return sum(len(tiler) for tiler in self._tilers)

    def __getitem__(self, key: ChipKey) -> np.ndarray:
        """Read ``(sample_index, tile_id)``; both indices are zero-based.

        The first index selects the source sample. The second is that sample's
        native Tiler tile ID. Returned pixels are independent copies.
        """
        sample_index, tile_id = key
        if sample_index < 0:
            raise IndexError("sample_index must be nonnegative")

        pixels = self._arrays[sample_index]
        tile = self._tilers[sample_index].get_tile(
            pixels.reshape((-1, *pixels.shape[-2:])), tile_id
        )
        return tile.reshape((*pixels.shape[:-2], *tile.shape[-2:]))

    def __iter__(self) -> Iterator[tuple[ChipKey, np.ndarray]]:
        """Yield ``((sample_index, tile_id), pixels)``; retain keys when batching."""
        for sample_index, tiler in enumerate(self._tilers):
            for tile_id in range(len(tiler)):
                key = (sample_index, tile_id)
                yield key, self[key]

    def merge(
        self,
        predictions: Iterable[tuple[ChipKey, np.ndarray]],
        *,
        window: Window | np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Merge predictions with native Tiler weighting in source order.

        Args:
            predictions: ``((sample_index, tile_id), pixels)`` pairs in any order.
                Spatial axes trail. Leading shapes must agree within a sample.
                Repeated tiles accumulate; missing tiles are allowed.
            window: Forwarded to Merger. Array windows must match its tile shape,
                with leading prediction axes flattened into one channel axis.

        Returns:
            One NumPy array per ``xarray`` source. Leading dimensions come from
            predictions; a source with no predictions uses its input shape.
            Native Merger normalization fills uncovered pixels with zero and
            handles nonfinite values. Output uses Merger's default float32 dtype.
            Each call uses fresh merger state.
        """
        mergers: dict[int, Merger] = {}
        shapes: dict[int, tuple[int, ...]] = {}

        for (sample_index, tile_id), pixels in predictions:
            if sample_index < 0:
                raise IndexError("sample_index must be nonnegative")
            layout = self._tilers[sample_index]
            shape = pixels.shape[:-2]
            if sample_index in shapes and shape != shapes[sample_index]:
                raise ValueError("prediction leading shape must agree within a sample")

            values = pixels.reshape((-1, *pixels.shape[-2:]))
            if sample_index not in mergers:
                # Output channels may differ from the source's input channels.
                tiler = Tiler(
                    data_shape=(values.shape[0], *layout.data_shape[-2:]),
                    tile_shape=(values.shape[0], *layout.tile_shape[-2:]),
                    channel_dimension=0,
                    overlap=layout.overlap,
                    mode=layout.mode,
                )
                mergers[sample_index] = Merger(tiler, window=window)
                shapes[sample_index] = shape

            mergers[sample_index].add(tile_id, values)

        outputs = []
        for index, layout in enumerate(self._tilers):
            if index in mergers:
                merger = mergers[index]
                shape = shapes[index]
            else:
                merger = Merger(layout, window=window)
                shape = self._arrays[index].shape[:-2]
            pixels = merger.merge()
            outputs.append(pixels.reshape((*shape, *pixels.shape[-2:])))
        return outputs

    @staticmethod
    def _validate_sample(sample: xr.Dataset | xr.DataArray) -> None:
        """Require shared dimensions and regular placement before conversion."""
        if isinstance(sample, xr.Dataset):
            sample.gs.validate(same_dims=True)
            if any(var.gs.geobox is not None for var in sample.data_vars.values()):
                sample.gs.validate(same_grid=True)
        else:
            sample.gs.validate()

        if sample.gs.geobox is not None and not isinstance(sample.gs.geobox, GeoBox):
            raise ValueError("Chips requires a regular grid or unreferenced pixels")
