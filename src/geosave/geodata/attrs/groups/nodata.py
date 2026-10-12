"""The stored value a variable's absent pixels hold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import AliasChoices, Field, model_validator

from xarray.core.utils import equivalent

from ..base import AttrsGroup, Decoded, Join, decode

if TYPE_CHECKING:
    import xarray as xr


class Nodata(AttrsGroup):
    """Name the stored value marking one variable's absent pixels.

    CF spells it `_FillValue` and odc `nodata`; both read, and only CF's is
    written, which odc and rioxarray read too.

    Args:
        value: Stored value marking absence.

    Raises:
        ValueError: The two spellings name different values, or the band's
            dtype cannot hold the value.

    Examples:
        >>> ds["B04"].gs.nodata
        0
    """

    JOIN: ClassVar[Join] = "agree"

    value: Decoded[int | float] = Field(
        validation_alias=AliasChoices("_FillValue", "nodata"),
        serialization_alias="_FillValue",
    )

    @model_validator(mode="before")
    @classmethod
    def _one_value(cls, data: Any) -> Any:
        """Refuse two spellings that name different values."""
        if not isinstance(data, dict) or not {"_FillValue", "nodata"} <= data.keys():
            return data
        cf, odc = decode(data["_FillValue"]), decode(data["nodata"])
        if not equivalent(cf, odc):
            raise ValueError(
                f"_FillValue {cf!r} and nodata {odc!r} disagree; "
                f"they spell one nodata, so write one with gs.write_nodata"
            )
        return data

    def held_by(self, band: xr.DataArray) -> Nodata:
        """Return the fill as `band` stores it.

        GDAL rounds a float fill to the band's precision, and CF masking
        compares in the band's dtype, so a float32 band reads 0.1 as float32's
        0.1. An integer band holds a fill exactly or not at all.

        Raises:
            ValueError: The fill overflows a float band, or an integer band
                would store another number, as -1 wraps to 65535 on uint16 and
                NaN has no integer.
        """
        with np.errstate(invalid="ignore", over="ignore"):
            stored = np.asarray(self.value).astype(band.dtype)
        if band.dtype.kind == "f":
            if np.isinf(stored) and not np.isinf(self.value):
                raise ValueError(f"{self.value!r} overflows a {band.dtype} band")
        elif not np.array_equal(stored, self.value):
            raise ValueError(
                f"{self.value!r} is no {band.dtype} value, so it marks no pixel of "
                f"the band; the band would store it as {stored.item()!r}"
            )
        return self.model_copy(update={"value": stored.item()})
