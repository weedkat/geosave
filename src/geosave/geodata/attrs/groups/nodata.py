"""The stored value a variable's absent pixels hold."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import AliasChoices, Field, model_validator

from xarray.core.utils import equivalent

from .base import AttrsGroup, Decoded, decode


class Nodata(AttrsGroup):
    """Name the stored value marking one variable's absent pixels.

    CF names it `_FillValue` and odc `nodata`; both aliases read, and only
    CF's is written, which odc and rioxarray read too.

    Attributes:
        fill_value: Stored value marking absence.

    Raises:
        ValueError: The attrs state `_FillValue` and `nodata` with different
            values.

    Examples:
        >>> ds["B04"].gs.nodata
        0
    """

    fill_value: Decoded[int | float] = Field(
        validation_alias=AliasChoices("_FillValue", "nodata"),
        serialization_alias="_FillValue",
    )

    def check_dtype(self, dtype: np.dtype) -> None:
        """Refuse a fill that `dtype` would store as another number.

        A pixel is masked when it equals the fill, so the fill must survive
        being stored. A float rounds it to its precision, which still matches
        its own pixels, but overflows a value past its range to inf. An integer
        stores it exactly or as another number: -1 wraps to 65535 on uint16,
        and NaN has no integer at all.

        Raises:
            ValueError: The fill does not fit `dtype`.
        """
        with np.errstate(invalid="ignore", over="ignore"):
            stored = np.asarray(self.fill_value).astype(dtype)
        if dtype.kind == "f":
            changed = bool(np.isinf(stored)) and not np.isinf(self.fill_value)
        else:
            changed = stored != self.fill_value
        if changed:
            raise ValueError(
                f"The nodata value {self.fill_value!r} does not fit this {dtype} band: "
                f"it would be stored as {stored.item()!r}, so the wrong pixels would "
                f"be masked. Choose a nodata value the band can hold, or convert the "
                f"band to a wider dtype first."
            )

    @model_validator(mode="before")
    @classmethod
    def _check_fill_values_consistent(cls, data: Any) -> Any:
        """Refuse attrs whose `_FillValue` and `nodata` hold different fill values."""
        if not isinstance(data, dict) or not {"_FillValue", "nodata"} <= data.keys():
            return data
        cf, odc = decode(data["_FillValue"]), decode(data["nodata"])
        if not equivalent(cf, odc):
            raise ValueError(
                f"The attrs give _FillValue as {cf!r} but nodata as {odc!r}, and both "
                f"name the same thing. Keep one of them, or set both to the same "
                f"value; gs.write_nodata(value) rewrites them consistently."
            )
        return data
