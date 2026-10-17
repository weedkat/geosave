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
        value: Stored value marking absence.

    Raises:
        ValueError: The two aliases name different values, or the band's
            dtype cannot hold the value.

    Examples:
        >>> ds["B04"].gs.nodata
        0
    """

    REFUSE_CONFLICT = True

    value: Decoded[int | float] = Field(
        validation_alias=AliasChoices("_FillValue", "nodata"),
        serialization_alias="_FillValue",
    )

    @model_validator(mode="before")
    @classmethod
    def _one_value(cls, data: Any) -> Any:
        """Refuse two aliases that name different values."""
        if not isinstance(data, dict) or not {"_FillValue", "nodata"} <= data.keys():
            return data
        cf, odc = decode(data["_FillValue"]), decode(data["nodata"])
        if not equivalent(cf, odc):
            raise ValueError(
                f"_FillValue {cf!r} and nodata {odc!r} disagree; "
                f"they name one nodata, so write one with gs.write_nodata"
            )
        return data

    def check_dtype(self, dtype: np.dtype) -> None:
        """Refuse a fill a variable of `dtype` cannot store as this number.

        A pixel is masked when it equals the fill, so the fill must survive
        being stored. A float variable rounds it to its precision, which still
        matches its own pixels, but overflows a value past its range to inf.
        An integer variable stores it exactly or as another number: -1 wraps
        to 65535 on uint16, and NaN has no integer at all.

        Args:
            dtype: Dtype of the variable the fill is written onto.

        Raises:
            ValueError: The variable would store another number.

        Examples:
            >>> Nodata(value=-1).check_dtype(np.dtype("uint16"))
            ValueError: -1 is no uint16 value; a uint16 variable stores it as 65535
        """
        with np.errstate(invalid="ignore", over="ignore"):
            stored = np.asarray(self.value).astype(dtype)
        if dtype.kind == "f":
            changed = bool(np.isinf(stored)) and not np.isinf(self.value)
        else:
            changed = stored != self.value
        if changed:
            raise ValueError(
                f"{self.value!r} is no {dtype} value; a {dtype} variable stores "
                f"it as {stored.item()!r}"
            )
