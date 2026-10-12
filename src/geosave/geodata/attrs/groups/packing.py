"""How a variable's stored digital numbers decode to physical values."""

from __future__ import annotations

from typing import ClassVar

from pydantic import field_validator

from .base import AttrsGroup, Decoded, Join


class Packing(AttrsGroup):
    """Describe how one variable's stored values decode to physical values.

    Rasters stay in their stored digital numbers in memory; decoding is the
    explicit `unpack`.

    Args:
        scale_factor: Multiplier applied to each stored value, never 0.
        add_offset: Value added after scaling.

    Examples:
        >>> ds["B04"].gs.packing
        Packing(scale_factor=0.0001, add_offset=-0.1)
    """

    JOIN: ClassVar[Join] = "agree"

    scale_factor: Decoded[float] = 1.0
    add_offset: Decoded[float] = 0.0

    @field_validator("scale_factor")
    @classmethod
    def _scales(cls, value: float) -> float:
        """Refuse a zero scale, which would decode every stored value to the offset."""
        if value == 0:
            raise ValueError("scale_factor 0 decodes every stored value alike")
        return value
