"""Cuts a model reads its rasters through: frames along time, chips in space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Self, cast

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from geosave.geodata.transform.extent import Padding
from geosave.geodata.core.vector.samples import Taper, Window

from .base import SpecModel

if TYPE_CHECKING:
    import geopandas as gpd
    from odc.geo.geobox import GeoBox

    from geosave.geodata import GeoDataFrame
    from geosave.geodata.core import GeoSamples


class FramesSpec(SpecModel):
    """Declare per-collection temporal bins required by a model."""

    windows: dict[str, Window]
    reference: str | None = None
    stride: PositiveInt | None = None

    def cut(self, samples: gpd.GeoDataFrame) -> GeoDataFrame:
        """Resolve temporal requirements within spatial sample rows."""
        sample_table = cast("GeoSamples", samples.gs)
        return sample_table.frame(
            windows=self.windows, reference=self.reference, stride=self.stride
        )


class ChipsSpec(SpecModel):
    """Declare how a raster is cut into chips and how results are merged back.

    Attributes:
        size: Chip side in pixels, or height and width.
        overlap: Pixels neighbouring chips share.
        mode: How a chip reaching past the raster is filled.
        window: How overlapping chips are weighed when merged. None weighs
            every chip alike.

    Raises:
        ValueError: Overlap cannot provide coverage for the selected window.

    Examples:
        >>> ChipsSpec(size=224, overlap=32).cut(samples).iloc[0]["proj:shape"]
        [224, 224]
    """

    size: PositiveInt | Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]
    overlap: NonNegativeInt = 0
    mode: Padding = "reflect"
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

    def cut(self, samples: gpd.GeoDataFrame, *, target: GeoBox) -> GeoDataFrame:
        """Cut samples into chips of this size.

        Args:
            samples: Spatial sample or frame table.
            target: Explicit pixel lattice used for training and inference.

        Returns:
            One row per chip, as `samples.gs.chip` lists them.
        """
        sample_table = cast("GeoSamples", samples.gs)
        return sample_table.chip(
            self.shape, target=target, overlap=self.overlap, pad_mode=self.mode
        )
