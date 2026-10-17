"""What an attr group is: a few attr keys that mean one value together.

Each group is a pydantic model over its keys; `from_attrs` reads one off an
object's attrs and `attrs.stamp_attrs` writes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Self

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
    pass


def decode(value: object) -> object:
    """Read GDAL text and NumPy scalars as native Python values.

    Args:
        value: Attribute value, optionally JSON text or a NumPy scalar.

    Returns:
        Parsed JSON value, a NumPy scalar's Python value, or the original
        value when neither conversion applies. Non-JSON strings stay strings.
    """
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

    Every subclass joins its parsed values through `combine_values`. Its keys
    are its fields' names, aliases, or alias choices.

    Attributes:
        KEYS: Every key the group reads; stamping replaces all of them.
        REFUSE_CONFLICT: Whether `combine_attrs` raises when every input
            states the group but not alike, as pixels with two nodata values
            mean different things. Otherwise it leaves the group out and warns.
            `intersect` never refuses.
    """

    # Groups are values; the other keys of an attrs mapping are none of theirs.
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    KEYS: ClassVar[tuple[str, ...]] = ()
    REFUSE_CONFLICT: ClassVar[bool] = False

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
    def is_stated_in(cls, attrs: Mapping[Any, Any]) -> bool:
        """Say whether the attrs state any key of the group.

        Args:
            attrs: Attribute mapping to inspect, including accepted aliases.

        Returns:
            True when at least one group key is present, even if its value
            would fail validation. No values are parsed.
        """
        return any(key in attrs for key in cls.KEYS)

    @classmethod
    def from_attrs(cls, attrs: Mapping[Any, Any]) -> Self | None:
        """Read the group from attrs, None where they state none of its keys.

        Args:
            attrs: Attribute mapping containing group fields or their aliases.
                Unrelated keys are ignored.

        Returns:
            Validated group, or None when no group key is present.

        Raises:
            ValueError: The attrs state the group invalidly.
        """
        return cls.model_validate(dict(attrs)) if cls.is_stated_in(attrs) else None

    @classmethod
    def replace(cls, attrs: Mapping[Any, Any], **fields: Any) -> Self:
        """Read the group from attrs with some fields changed.

        Args:
            attrs: Attribute mapping stating the group, or none of it.
            **fields: Field names mapped to their new values; None clears one.

        Returns:
            The changed group; one whose fields are all unset writes no attrs.

        Raises:
            ValueError: The changed group is invalid.

        Examples:
            >>> attrs.stamp_attrs(band, GDALBand.replace(band.attrs, colorinterp="red"))
        """
        stated = cls.from_attrs(attrs)
        current = {} if stated is None else stated.model_dump()
        return cls.model_validate(current | fields)

    @classmethod
    def intersect_values(cls, values: Sequence[Self | None]) -> Self | None:
        """Keep the value every object states alike.

        Override it where a group's parts are independent facts.

        Args:
            values: Each object's value for the group, None where it states none.

        Returns:
            The shared value, or None where any object states none or they differ.
        """
        first = values[0]
        return first if all(value == first for value in values) else None

    @classmethod
    def combine_values(cls, values: Sequence[Self | None]) -> Self | None:
        """Join parsed values, refusing conflicts according to the group's policy.

        Args:
            values: Each input's group, None where it states none.

        Returns:
            Shared value, or None when absent or different across inputs.

        Raises:
            ValueError: `REFUSE_CONFLICT` is set and every input states a different value.
        """
        joined = cls.intersect_values(values)
        if cls.REFUSE_CONFLICT and None not in values and joined is None:
            stated = sorted({repr(value) for value in values})
            raise ValueError(
                f"the inputs joined state {cls.__name__} differently: {stated}; "
                f"align it before joining them"
            )
        return joined

    def to_attrs(self) -> dict[str, Any]:
        """Return the attrs this value writes; a field at its default or None writes none.

        netCDF refuses a None attr, so an unset field leaves its key out.

        Returns:
            New mapping using serialization aliases and JSON-compatible
            values, omitting fields at their defaults or set to None.
        """
        return self.model_dump(
            by_alias=True, mode="json", exclude_defaults=True, exclude_none=True
        )

    def __eq__(self, other: object) -> bool:
        """Compare the attrs each group writes, treating NaN as equal to NaN.

        Args:
            other: Value to compare with this group.

        Returns:
            True when both groups have the same concrete type and serialize
            equivalent attrs; False when those attrs differ. Returns
            NotImplemented for another type so Python can try its comparison.
        """
        if type(other) is not type(self):
            return NotImplemented
        mine, theirs = self.to_attrs(), other.to_attrs()
        return mine.keys() == theirs.keys() and all(
            equivalent(mine[key], theirs[key]) for key in mine
        )

    __hash__ = BaseModel.__hash__
