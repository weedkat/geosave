"""Lay fixed-size chips over raster grids, without reading a pixel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as to_integer
from typing import Literal

from affine import Affine
from odc.geo.geobox import GeoBox

# How chips meet a raster's edge: reaching past it and filled as `numpy.pad`
# fills, moved inward to sit flush with it, or left out when incomplete.
type ChipMode = Literal["reflect", "edge", "wrap", "constant", "shift", "drop"]

_CHIP_MODES = ("reflect", "edge", "wrap", "constant", "shift", "drop")


@dataclass(frozen=True, slots=True)
class ChipLocation:
    """Where one chip sits: its raster, and the pixel of its top-left corner.

    Attributes:
        raster_index: Position of the chip's raster in `ChipLayout.geoboxes`.
        top: Row of the chip's first pixel in that raster.
        left: Column of the chip's first pixel in that raster.
    """

    raster_index: int
    top: int
    left: int


def lay_chip_offsets(
    length: int, *, size: int, stride: int, mode: ChipMode
) -> list[int]:
    """Lay chips along one axis and return each one's first pixel.

    Args:
        length: Pixels along the axis.
        size: Pixels of one chip.
        stride: Pixels between the starts of consecutive chips.
        mode: How the last chip meets the end of the axis.

    Returns:
        First pixel of each chip, ascending. A padding mode steps until the
        axis is covered, so the last chip may reach past the end; `"shift"`
        adds a last chip flush with the end; `"drop"` keeps only whole chips,
        possibly none.

    Raises:
        ValueError: `"shift"` is asked of an axis shorter than one chip.

    Examples:
        >>> lay_chip_offsets(10, size=4, stride=4, mode="reflect")
        [0, 4, 8]
        >>> lay_chip_offsets(10, size=4, stride=4, mode="shift")
        [0, 4, 6]
        >>> lay_chip_offsets(10, size=4, stride=4, mode="drop")
        [0, 4]
    """
    if mode == "drop":
        return list(range(0, length - size + 1, stride))

    if mode == "shift":
        if length < size:
            raise ValueError(
                f"the raster spans {length} pixels, fewer than one chip of "
                f"{size}; use a smaller size or a padding mode"
            )
        offsets = list(range(0, length - size + 1, stride))
        if offsets[-1] != length - size:
            offsets.append(length - size)
        return offsets

    offsets = [0]
    while offsets[-1] + size < length:
        offsets.append(offsets[-1] + stride)
    return offsets


class ChipLayout:
    """Where every chip sits on a sequence of raster grids.

    Chips are numbered raster by raster, row by row. The layout holds grids
    only, so it is what a `RasterChipper` cuts by, what a `RasterMerger` places
    predictions by, and what a stored set of chips needs to be merged later
    without its rasters.

    Args:
        geoboxes: Pixel grid of each raster; they may differ in grid and size.
        size: Chip height and width in pixels.
        stride: Pixels between chip starts; None takes `size`, so chips do
            not overlap.
        mode: How chips meet a raster's edge. `"reflect"`, `"edge"`, `"wrap"`
            and `"constant"` let the last chips reach past the edge;
            `"shift"` moves the last chips inward, flush with the edge;
            `"drop"` leaves incomplete chips out.

    Raises:
        ValueError: `size` or `stride` is below 1, `mode` is unknown, or
            `"shift"` meets a raster smaller than one chip.

    Examples:
        >>> layout = ChipLayout([scene.gs.geobox], size=256, stride=224)
        >>> layout.locate(1)
        ChipLocation(raster_index=0, top=0, left=224)
        >>> layout.geobox(1).shape
        Shape2d(x=256, y=256)
    """

    def __init__(
        self,
        geoboxes: Sequence[GeoBox],
        *,
        size: int,
        stride: int | None = None,
        mode: ChipMode = "reflect",
    ) -> None:
        if stride is None:
            stride = size
        if size < 1 or stride < 1:
            raise ValueError("chip size and stride must be at least 1")
        if mode not in _CHIP_MODES:
            raise ValueError(f"mode must be one of {_CHIP_MODES}, got {mode!r}")

        self.geoboxes = tuple(geoboxes)
        self.size = size
        self.stride = stride
        self.mode: ChipMode = mode

        self._locations: list[ChipLocation] = []
        self._chip_counts: list[int] = []
        for raster_index, geobox in enumerate(self.geoboxes):
            locations = self._lay_chips(raster_index, geobox)
            self._locations.extend(locations)
            self._chip_counts.append(len(locations))

    def _lay_chips(self, raster_index: int, geobox: GeoBox) -> list[ChipLocation]:
        """Lay one raster's chips, row by row."""
        tops = lay_chip_offsets(
            geobox.shape.y, size=self.size, stride=self.stride, mode=self.mode
        )
        lefts = lay_chip_offsets(
            geobox.shape.x, size=self.size, stride=self.stride, mode=self.mode
        )
        locations = []
        for top in tops:
            for left in lefts:
                locations.append(ChipLocation(raster_index, top, left))
        return locations

    def __len__(self) -> int:
        """Return the number of chips across every raster."""
        return len(self._locations)

    def count_chips(self, raster_index: int) -> int:
        """Return the number of chips laid over one raster.

        Raises:
            IndexError: The layout has no such raster.
        """
        return self._chip_counts[raster_index]

    def locate(self, index: int) -> ChipLocation:
        """Return a chip's raster and corner pixel.

        Raises:
            IndexError: The chip index is outside the chips.
        """
        index = to_integer(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("chip index out of range")
        return self._locations[index]

    def geobox(self, index: int) -> GeoBox:
        """Return a chip's pixel grid, padded part included.

        Raises:
            IndexError: The chip index is outside the chips.
        """
        location = self.locate(index)
        scene = self.geoboxes[location.raster_index]
        # A chip may reach past the raster, where slicing its GeoBox cannot go.
        transform = scene.transform * Affine.translation(location.left, location.top)
        return GeoBox((self.size, self.size), transform, scene.crs)
