"""How a variable's stored digital numbers decode to physical values."""

from __future__ import annotations

from pydantic import field_validator

from .base import AttrsGroup, Decoded


class Packing(AttrsGroup):
    """State how one variable's stored values decode to physical values.

    Rasters stay in their stored digital numbers in memory; decoding is the
    explicit `unpack`.

    Attributes:
        scale_factor: Multiplier applied to each stored value, never 0.
        add_offset: Value added after scaling.

    Examples:
        >>> ds["B04"].gs.packing
        Packing(scale_factor=0.0001, add_offset=-0.1)
    """

    scale_factor: Decoded[float] = 1.0
    add_offset: Decoded[float] = 0.0

    @field_validator("scale_factor")
    @classmethod
    def _check_scale_factor_not_zero(cls, scale_factor: float) -> float:
        """Refuse a zero scale factor, which decodes every stored value to the offset."""
        if scale_factor == 0:
            raise ValueError(
                "scale_factor cannot be 0, because every stored value would decode "
                "to the same number. Use the multiplier that converts stored values "
                "to physical ones, such as 0.0001 for Sentinel-2 reflectance."
            )
        return scale_factor
