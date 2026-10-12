"""What an attr group is: a few attr keys that mean one value together.

Each group is a pydantic model over its keys; `attrs.parse` reads one off a
band and `attrs.stamp` writes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

import numpy as np
import orjson
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
)
from xarray.core.utils import equivalent

if TYPE_CHECKING:
    import xarray as xr

type Join = Literal["agree", "drop", "each"]


def decode(value: object) -> object:
    """Read GDAL's text spelling and NumPy scalars as the native values they spell."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError:
            return value
    return value


# GDAL tags are text; Zarr, NetCDF and memory are native. Both read the same.
type Decoded[T] = Annotated[T, BeforeValidator(decode)]


class AttrsGroup(BaseModel):
    """A set of attr keys read and written as one validated value.

    A subclass is joined by `combine` as soon as it is defined. Its keys are
    its fields' names, aliases, or alias choices.

    Attributes:
        KEYS: Every key the group reads; stamping replaces all of them.
        JOIN: What a join does when objects state different values:
            `"agree"` refuses it, `"drop"` leaves the group out, `"each"`
            joins every key on its own, as for facts that do not depend on
            each other.
    """

    # Groups are values; the other keys of an attrs mapping are none of theirs.
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    KEYS: ClassVar[tuple[str, ...]] = ()
    JOIN: ClassVar[Join]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        keys: list[str] = []
        for name, field in cls.model_fields.items():
            if isinstance(field.validation_alias, AliasChoices):
                keys += [str(choice) for choice in field.validation_alias.choices]
            else:
                keys.append(field.alias or name)
        cls.KEYS = tuple(keys)

    @classmethod
    def stated_in(cls, attrs: Mapping[Any, Any]) -> bool:
        """Say whether the attrs state any key of the group."""
        return any(key in attrs for key in cls.KEYS)

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> Self | None:
        """Read the group from attrs, None where they state none of its keys.

        Raises:
            ValueError: The attrs state the group invalidly.
        """
        return cls.model_validate(dict(attrs)) if cls.stated_in(attrs) else None

    def held_by(self, band: xr.DataArray) -> Self:
        """Return the value as `band` stores it; most groups name no pixel value.

        Raises:
            ValueError: The band's dtype cannot hold the value.
        """
        return self

    def to_attrs(self) -> dict[str, Any]:
        """Return the attrs this value writes; a field at its default or None writes none.

        netCDF refuses a None attr, so an unset field leaves its key out.
        """
        return self.model_dump(
            by_alias=True, mode="json", exclude_defaults=True, exclude_none=True
        )

    def __eq__(self, other: object) -> bool:
        """Compare by the attrs each writes, as xarray compares attrs: NaN equals NaN."""
        if type(other) is not type(self):
            return NotImplemented
        mine, theirs = self.to_attrs(), other.to_attrs()
        return mine.keys() == theirs.keys() and all(
            equivalent(mine[key], theirs[key]) for key in mine
        )

    __hash__ = BaseModel.__hash__
