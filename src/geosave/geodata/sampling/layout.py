"""Where chips sit on their rasters, and how chips are laid along an axis."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import accumulate
from numbers import Integral, Real
from typing import Literal

from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from geosave.geodata.utils.datetime import DateRange

# How chips meet a raster's edge: reaching past it and filled as `numpy.pad`
# fills, moved inward to sit flush with it, or left out when incomplete.
type ChipMode = Literal["reflect", "edge", "wrap", "constant", "shift", "drop"]

CHIP_MODES = ("reflect", "edge", "wrap", "constant", "shift", "drop")


@dataclass(frozen=True, slots=True)
class ChipSource:
    """The raster chips are cut from: its position in the input, grid, dates and extent.

    One instance is shared by all of a raster's chips. It compares by index,
    grid and dates, so a copy made by pickling, such as through a DataLoader
    worker, still names the same raster.

    Attributes:
        index: Position of the raster or temporal window in the chipper's
            input, from 0.
        geobox: The raster's grid.
        timespan: The time the raster covers, or None for a timeless raster.
        extent: The place the raster declares it covers, which a merger
            writes back onto its predictions; None when it declares none.
    """

    index: int
    geobox: GeoBox
    timespan: DateRange | None
    extent: Geometry | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class ChipLocation:
    """Where one chip sits: its raster, and the pixel of its top-left corner.

    Attributes:
        source: The raster the chip was cut from.
        top: Row of the chip's first pixel in that raster.
        left: Column of the chip's first pixel in that raster.
    """

    source: ChipSource
    top: int
    left: int

    def geobox(self, size: int) -> GeoBox:
        """Return the chip's own grid, padded part included."""
        grid = self.source.geobox
        transform = grid.transform * Affine.translation(self.left, self.top)
        return GeoBox((size, size), transform, grid.crs)


@dataclass(frozen=True, slots=True)
class ChipBlock:
    """Raster pixels read in one computation, and the chips cut from them.

    Attributes:
        rows: The block's rows in the grid every chip lies on.
        columns: The block's columns in that grid.
        locations: The chips lying wholly inside the block, row by row.
    """

    rows: slice
    columns: slice
    locations: list[ChipLocation]

    def find_chip_pixels(
        self, location: ChipLocation, size: int
    ) -> tuple[slice, slice]:
        """Return the rows and columns of one of the block's chips, within the block."""
        top = location.top - self.rows.start
        left = location.left - self.columns.start
        return slice(top, top + size), slice(left, left + size)


def lay_chips_geobox(chip_rows: list[list[ChipLocation]], size: int) -> GeoBox:
    """Return the grid every chip lies on: the raster's, extended where chips reach past its edge.

    Args:
        chip_rows: One raster's chip locations, row by row.
        size: Chip height and width in pixels.

    Returns:
        The raster's grid from its first pixel to the far edge of the last chip.
    """
    raster_geobox = chip_rows[0][0].source.geobox
    last_chip = chip_rows[-1][-1]
    shape = (last_chip.top + size, last_chip.left + size)
    return GeoBox(shape, raster_geobox.transform, raster_geobox.crs)


def find_block_ends(
    chunk_lengths: tuple[int, ...] | None, length: int, min_length: int
) -> list[int]:
    """Return where blocks end along one axis: at Dask chunk ends, `min_length` apart or more.

    A block spanning several chunks leaves fewer chips crossing its edge, and
    those chips are the ones that read a chunk again.

    Args:
        chunk_lengths: Chunk lengths along the axis, or None for pixels in memory.
        length: Pixels along the axis.
        min_length: Fewest pixels a block spans, except the last.

    Returns:
        Block ends, ascending, the last at `length`.

    Examples:
        >>> find_block_ends((256, 256, 256, 256, 102), 1126, min_length=896)
        [1024, 1126]
        >>> find_block_ends(None, 1126, min_length=896)
        [1126]
    """
    if chunk_lengths is None:
        return [length]

    block_ends = []
    block_start = 0
    for chunk_end in accumulate(chunk_lengths):
        if chunk_end - block_start >= min_length or chunk_end == length:
            block_ends.append(chunk_end)
            block_start = chunk_end
    return block_ends


def lay_chip_blocks(
    chip_rows: list[list[ChipLocation]],
    block_row_ends: list[int],
    block_column_ends: list[int],
    size: int,
) -> list[ChipBlock]:
    """Group chips into blocks, each chip in the first block that holds it whole.

    A block starts at its first chip and ends at a block end, so it spans at
    most one block length and one chip in each direction.

    Args:
        chip_rows: One raster's chip locations, row by row.
        block_row_ends: Where blocks end down the rows, as `find_block_ends`
            returns them; the last is at least the lower edge of the last chip.
        block_column_ends: Where blocks end across the columns; the last is
            at least the right edge of the last chip.
        size: Chip height and width in pixels.

    Returns:
        Blocks in raster order, together holding every chip once.

    Examples:
        >>> blocks = lay_chip_blocks(chip_rows, [512, 1064], [512, 1064], size=224)
        >>> blocks[1].rows, blocks[1].columns
        (slice(0, 512, None), slice(336, 1064, None))
    """
    blocks = []
    pending_rows = list(chip_rows)
    for block_row_end in block_row_ends:
        # Chip rows ending in these block rows; later block rows read the rest.
        strip_locations = []
        while pending_rows and pending_rows[0][0].top + size <= block_row_end:
            strip_locations.extend(pending_rows.pop(0))

        for block_column_end in block_column_ends:
            locations = []
            later_locations = []
            for location in strip_locations:
                if location.left + size <= block_column_end:
                    locations.append(location)
                else:
                    later_locations.append(location)
            strip_locations = later_locations
            if not locations:
                continue

            top = min(location.top for location in locations)
            left = min(location.left for location in locations)
            rows = slice(top, block_row_end)
            columns = slice(left, block_column_end)
            blocks.append(ChipBlock(rows, columns, locations))
    return blocks


def resolve_overlap(size: int, overlap: int | float) -> int:
    """Resolve a pixel count or a fraction of the chip width to whole pixels."""
    if size < 1:
        raise ValueError("chip size must be at least 1")
    if isinstance(overlap, bool) or not isinstance(overlap, Real):
        raise TypeError("overlap must be an integer pixel count or a float fraction")
    if isinstance(overlap, Integral):
        if not 0 <= overlap < size:
            raise ValueError(f"pixel overlap must be between 0 and {size - 1}")
        return int(overlap)
    if not 0 <= overlap < 1:
        raise ValueError("fractional overlap must be in [0, 1)")
    return int(size * overlap)


def lay_chip_offsets(
    length: int, *, size: int, overlap: int | float = 0, mode: ChipMode
) -> list[int]:
    """Lay chips along one axis and return each one's first pixel.

    Args:
        length: Pixels along the axis.
        size: Pixels of one chip.
        overlap: Shared pixels per axis, or a fraction of `size` rounded down.
        mode: How the last chip meets the end of the axis.

    Returns:
        First pixel of each chip, ascending. A padding mode steps until the
        axis is covered, so the last chip may reach past the end; `"shift"`
        adds a last chip flush with the end; `"drop"` keeps only whole chips,
        possibly none.

    Raises:
        ValueError: `"shift"` is asked of an axis shorter than one chip.

    Examples:
        >>> lay_chip_offsets(10, size=4, mode="reflect")
        [0, 4, 8]
        >>> lay_chip_offsets(10, size=4, mode="shift")
        [0, 4, 6]
        >>> lay_chip_offsets(10, size=4, mode="drop")
        [0, 4]
    """
    step = size - resolve_overlap(size, overlap)
    if mode not in CHIP_MODES:
        raise ValueError(f"mode must be one of {CHIP_MODES}, got {mode!r}")
    if mode == "drop":
        return list(range(0, length - size + 1, step))

    if mode == "shift":
        if length < size:
            raise ValueError(
                f"the raster spans {length} pixels, fewer than one chip of "
                f"{size}; use a smaller size or a padding mode"
            )
        offsets = list(range(0, length - size + 1, step))
        if offsets[-1] != length - size:
            offsets.append(length - size)
        return offsets

    # Include the first chip that reaches or passes the far edge.
    return list(range(0, max(0, length - size) + step, step))
