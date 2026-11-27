"""Iterate over model-sized chips of prepared rasters."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, Any, overload

from odc.geo.geobox import GeoBox

from geosave.geodata.core.raster.transform.extent import Padding
from geosave.geodata.utils.rasters import Raster, as_rasters

from .layout import (
    CHIP_MODES,
    ChipLocation,
    ChipMode,
    ChipSource,
    find_block_ends,
    lay_chip_blocks,
    lay_chip_offsets,
    lay_chips_geobox,
    resolve_overlap,
)

if TYPE_CHECKING:
    from geosave.geodata.core.raster.utils.context import ContextReader, ContextTensors


# A block at least four chips across leaves about one chip in four per axis
# crossing its edge and reading a chunk again, while staying a few chunks large.
CHIPS_PER_BLOCK = 4


@dataclass(frozen=True)
class ChipBatch[T: Raster]:
    """Chips read together from their rasters: where each sits, and its pixels in memory.

    Attributes:
        locations: Each chip's placement, in batch order.
        chips: Each chip as its raster's xarray type, loaded, in batch order.

    Examples:
        >>> pixels, context = batch.to_tensor(context={"centroid": "centroid"})
        >>> logits = model(image=pixels, **context)
        >>> outputs = zip(batch.locations, logits.numpy())
    """

    locations: tuple[ChipLocation, ...]
    chips: tuple[T, ...]

    @overload
    def to_tensor(self, *, context: None = None) -> Any: ...

    @overload
    def to_tensor(
        self, *, context: Mapping[str, ContextReader[T] | str]
    ) -> tuple[Any, ContextTensors]: ...

    def to_tensor(
        self, *, context: Mapping[str, ContextReader[T] | str] | None = None
    ) -> Any | tuple[Any, ContextTensors]:
        """Stack each chip's `chip.gs.to_tensor(context=context)` along a new batch axis.

        Args:
            context: Model input names mapped to readers, as for the chips'
                `gs.to_tensor`. None reads no context.

        Returns:
            Pixels shaped `(batch, ...)`: one tensor for DataArray and Dataset
            chips, one per raster node, keyed by its path, for DataTree chips;
            with `context`, the pixels and the stacked context tensors.

        Raises:
            KeyError: A reader name is not in `CONTEXT_READERS`.
        """
        from torch.utils.data import default_collate

        if context is None:
            return default_collate([chip.gs.to_tensor() for chip in self.chips])

        samples = [chip.gs.to_tensor(context=context) for chip in self.chips]
        pixels, context_tensors = default_collate(samples)
        return pixels, context_tensors


class RasterChipper[T: Raster]:
    """Yield each chip's placement and cropped xarray object in raster order.

    Every chip's location names its raster as its source, so a merger built
    from the chipper places predictions from mixed or shuffled batches back
    on the raster they came from. The rasters are held as given, lazy;
    Dask-backed pixels stay lazy until a chip is read, including at padded
    edges.

    Args:
        rasters: One DataArray, Dataset or DataTree, or an iterable of them.
            Each tree's raster leaves must share a grid.
        size: Chip height and width in pixels.
        overlap: An integer number of shared pixels per axis, or a float
            fraction of the chip size in [0, 1), rounded down to whole pixels.
            For size 256, both 64 and 0.25 advance by 192 pixels.
        mode: Edge policy. Shift moves the last chip inward and can increase
            its overlap. Drop skips incomplete chips. Padding fills the border.
        constant_value: Fill for constant padding.

    Attributes:
        rasters: The input rasters, in input order.
        sources: Each raster's grid, timespan and extent, in input order.

    Raises:
        RasterValidationError: An input lacks a georeferenced grid.

    Examples:
        >>> chips = gs.RasterChipper(scenes, size=256, overlap=0.25)
        >>> merger = gs.RasterMerger(chips, window="hann")
        >>> for location, chip in chips:
        ...     merger.add(location, predict(chip))
        >>> scores = merger.merge()  # one DataArray per scene

        Model input as tensors, read one block of chunks at a time:

        >>> for batch in chips.iter_batches(16):
        ...     pixels, context = batch.to_tensor(context={"centroid": "centroid"})
        ...     logits = model(image=pixels, **context)
    """

    def __init__(
        self,
        rasters: T | Iterable[T],
        *,
        size: int,
        overlap: int | float = 0,
        mode: ChipMode = "reflect",
        constant_value: int | float = 0,
    ) -> None:
        if mode not in CHIP_MODES:
            raise ValueError(f"mode must be one of {CHIP_MODES}, got {mode!r}")
        self.size = size
        self.overlap = resolve_overlap(size, overlap)
        self.mode: ChipMode = mode
        self.constant_value = constant_value
        # Shifted and dropped chips never leave the raster; pad them as constant.
        self._padding: Padding = "constant" if mode in ("shift", "drop") else mode
        self.rasters = tuple(as_rasters(rasters))
        sources = []
        for index, raster in enumerate(self.rasters):
            raster.gs.validate()
            sources.append(
                ChipSource(index, raster.gs.geobox, raster.gs.timespan, raster.gs.extent)
            )
        self.sources = tuple(sources)

    def __iter__(self) -> Iterator[tuple[ChipLocation, T]]:
        """Cut each chip lazily, row by row, preserving its raster's xarray type.

        Each chip is padded alone, so it reads only its own chunks.
        """
        for source, raster in zip(self.sources, self.rasters, strict=True):
            for row in self._lay_rows(source):
                for location in row:
                    yield location, self._pad(raster, location.geobox(self.size))

    def iter_batches(self, batch_size: int) -> Iterator[ChipBatch[T]]:
        """Yield the chips in batches, each chip loaded, block by block.

        The raster is read one block of Dask chunks at a time, and each chip
        is cut from the block in memory, so a chunk is read once, or again by
        chips reaching past its lower or right edge. The chips equal those
        `iter(chipper)` yields, loaded, in block order; a batch can span
        rasters, and the last holds the remaining chips.

        Holds one block in memory: every band and frame of the input over at
        most one chunk plus `size` pixels in each direction. A raster without
        Dask chunks is read whole.

        Args:
            batch_size: Chips per batch.

        Yields:
            Each batch's chip locations and loaded chips.

        Examples:
            >>> for batch in chips.iter_batches(16):
            ...     pixels, context = batch.to_tensor(context={"centroid": "centroid"})
            ...     logits = model(image=pixels, **context)
        """
        for group in batched(self._read_chips(), batch_size):
            locations, chips = zip(*group, strict=True)
            yield ChipBatch(locations, chips)

    def _read_chips(self) -> Iterator[tuple[ChipLocation, T]]:
        """Load each block in one computation and slice its chips from memory."""
        min_length = CHIPS_PER_BLOCK * self.size
        for source, raster in zip(self.sources, self.rasters, strict=True):
            chip_rows = self._lay_rows(source)
            if not chip_rows:
                continue

            chips_geobox = lay_chips_geobox(chip_rows, self.size)
            padded_raster = self._pad(raster, chips_geobox)
            ydim, xdim = padded_raster.gs.spatial_dims
            blocks = lay_chip_blocks(
                chip_rows,
                find_block_ends(
                    padded_raster.gs.chunk_heights, chips_geobox.height, min_length
                ),
                find_block_ends(
                    padded_raster.gs.chunk_widths, chips_geobox.width, min_length
                ),
                self.size,
            )

            for block in blocks:
                pixels = padded_raster.isel({ydim: block.rows, xdim: block.columns})
                pixels = pixels.compute()
                for location in block.locations:
                    rows, columns = block.find_chip_pixels(location, self.size)
                    yield location, pixels.isel({ydim: rows, xdim: columns})

    def _pad(self, raster: T, geobox: GeoBox) -> T:
        """Pad lazily once, so every chip is a slice of the padded raster."""
        return raster.gs.crop_to_grid(
            geobox, mode=self._padding, constant_value=self.constant_value
        )

    def _lay_rows(self, source: ChipSource) -> list[list[ChipLocation]]:
        """Lay one raster's chips, row by row, naming the raster as their source."""
        grid = source.geobox
        tops = lay_chip_offsets(
            grid.height, size=self.size, overlap=self.overlap, mode=self.mode
        )
        lefts = lay_chip_offsets(
            grid.width, size=self.size, overlap=self.overlap, mode=self.mode
        )

        rows = []
        # Dropping incomplete chips can leave a raster narrower than one chip with none.
        if not lefts:
            return rows

        for top in tops:
            row = []
            for left in lefts:
                row.append(ChipLocation(source, top, left))
            rows.append(row)
        return rows
