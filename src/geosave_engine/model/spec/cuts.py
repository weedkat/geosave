"""Cuts a model reads its rasters through: frames along time, chips in space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from geosave_engine.geodata import cuts
from geosave_engine.geodata.cuts.chips import ChipMode
from geosave_engine.geodata.cuts.frames import FrameMode
from geosave_engine.geodata.cuts.merge import Taper

from .base import SpecModel, Text

if TYPE_CHECKING:
    import geopandas as gpd
    from tiler import Tiler

    from geosave_engine.geodata import GeoDataFrame


class FramesSpec(SpecModel):
    """Declare how a sample's time axis is cut into frames.

    Args:
        length: Instants each frame holds.
        stride: Instants between the starts of consecutive frames. None
            strides by `length`.
        tolerance: How far an instant may reach for a scene, such as `"10D"`.
        mode: What a frame covering an instant some layer missed does.

    Examples:
        >>> FramesSpec(length=4, stride=2, tolerance="10D").cut(windows)["parent"].tolist()
        [0, 0]
    """

    length: PositiveInt
    stride: PositiveInt | None = None
    tolerance: Text
    mode: FrameMode = "strict"

    def cut(self, windows: gpd.GeoDataFrame) -> GeoDataFrame:
        """Cut windows into frames, each collection keeping its own dates.

        Args:
            windows: Windows as `cuts.stacks` or `cuts.chips` lists them.

        Returns:
            One row per frame, as `cuts.frames` lists them.
        """
        return cuts.frames(
            windows,
            self.length,
            tolerance=self.tolerance,
            stride=self.stride,
            mode=self.mode,
        )


class ChipsSpec(SpecModel):
    """Declare how a raster is cut into chips and how results are merged back.

    Args:
        size: Chip side in pixels, or height and width.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past the raster is filled.
        window: How overlapping chips are weighed when merged. None weighs
            every chip alike.

    Raises:
        ValueError: Overlap cannot provide coverage for the selected window.

    Examples:
        >>> ChipsSpec(size=224, overlap=32).cut(windows).iloc[0][["parent", "chip"]].tolist()
        [0, 0]
    """

    size: PositiveInt | Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]
    overlap: NonNegativeInt = 0
    mode: ChipMode = "reflect"
    window: Taper | None = None

    @model_validator(mode="after")
    def _validate_window(self) -> Self:
        if self.window is not None and not self.overlap:
            raise ValueError(
                f"window {self.window!r} weighs overlapping chips; set overlap"
            )
        # Zero or negative edge weights need two pixels of shared support.
        if self.overlap == 1 and self.window in (
            "hann",
            "bartlett",
            "barthann",
            "bohman",
            "blackman",
            "overlap-tile",
        ):
            raise ValueError(
                f"window {self.window!r} needs at least 2 pixels of overlap to cover seams"
            )
        return self

    @property
    def shape(self) -> tuple[int, int]:
        """Return the chip height and width."""
        if isinstance(self.size, int):
            return self.size, self.size
        height, width = self.size
        return height, width

    def cut(self, windows: gpd.GeoDataFrame) -> GeoDataFrame:
        """Cut windows into chips of this size.

        Args:
            windows: Windows as `cuts.stacks` or `cuts.frames` lists them.

        Returns:
            One row per chip, as `cuts.chips` lists them.
        """
        return cuts.chips(windows, self.shape, overlap=self.overlap, mode=self.mode)

    def layout(self, shape: tuple[int, int]) -> tuple[Tiler, list[tuple[int, int]]]:
        """Lay these chips over a window, as cutting them did.

        Args:
            shape: Height and width of the window.

        Returns:
            The Tiler laying the chips, and the halo before and after each axis.
        """
        return cuts.layout(shape, self.shape, overlap=self.overlap, mode=self.mode)
