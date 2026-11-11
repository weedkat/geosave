"""CF scale, offset and missing-value declarations for one variable."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import AliasChoices, Field, field_validator, model_validator
from xarray.core.utils import equivalent

from .base import AttrsGroup, Decoded, decode


class Packing(AttrsGroup):
    """State how one variable's stored values decode to physical values.

    These fields live in attrs while pixels hold stored values. Native CF
    decoding moves consumed fields into encoding for the reverse conversion.

    Attributes:
        scale_factor: Multiplier applied to each stored value, never 0.
        add_offset: Value added after scaling.
        fill_value: Stored missing-value sentinel, written as `_FillValue`.
            ODC's `nodata` spelling is also accepted when reading.
        missing_value: Additional CF missing-value code or codes.

    Examples:
        >>> ds["B04"].gs.packing.scale_factor
        0.0001
    """

    scale_factor: Decoded[float] = 1.0
    add_offset: Decoded[float] = 0.0
    fill_value: Decoded[int | float] | None = Field(
        default=None,
        validation_alias=AliasChoices("_FillValue", "nodata"),
        serialization_alias="_FillValue",
    )
    missing_value: Decoded[int | float | list[int | float]] | None = None

    def check_dtype(self, dtype: np.dtype) -> None:
        """Refuse missing-value codes the variable's dtype cannot represent."""
        values = []
        if self.fill_value is not None:
            values.append(self.fill_value)
        if self.missing_value is not None:
            values.extend(np.atleast_1d(self.missing_value).tolist())

        for value in values:
            with np.errstate(invalid="ignore", over="ignore"):
                stored = np.asarray(value).astype(dtype)
            if dtype.kind == "f":
                changed = bool(np.isinf(stored)) and not np.isinf(value)
            else:
                changed = stored != value
            if changed:
                raise ValueError(
                    f"The nodata value {value!r} does not fit this {dtype} band: "
                    f"it would be stored as {stored.item()!r}. Choose a value "
                    "the band can hold, or convert to a wider dtype first."
                )

    @model_validator(mode="before")
    @classmethod
    def _check_fill_aliases(cls, data: Any) -> Any:
        """Reject disagreeing CF and ODC spellings of the same sentinel."""
        if isinstance(data, dict) and {"_FillValue", "nodata"} <= data.keys():
            cf, odc = decode(data["_FillValue"]), decode(data["nodata"])
            if not equivalent(cf, odc):
                raise ValueError(
                    f"The attrs give _FillValue as {cf!r} but nodata as {odc!r}; "
                    "both must describe the same missing-value code."
                )
        return data

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
