"""Validated input and step declarations for model workflows."""

import keyword
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    field_validator,
    model_validator,
)

from .reference import Reference, find_references


class Input(BaseModel):
    """Declare an external input and its optional fallback value.

    An omitted default means the caller must supply the input. An explicitly
    declared null default supplies None. Native values are not converted into
    configuration models.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    default: object = None

    @model_validator(mode="after")
    def check_default(self) -> Self:
        """Keep input defaults independent of runtime references."""
        if find_references(self.default):
            raise ValueError("Input defaults cannot contain references")
        return self


class Step(BaseModel):
    """Declare an operation and its literal or referenced keyword arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)

    call: str | InstanceOf[Reference]
    kwargs: dict[str, object] = Field(default_factory=dict)

    @field_validator("call")
    @classmethod
    def check_operation(cls, value: str | Reference) -> str | Reference:
        """Check an operation's name without importing or evaluating it."""
        if isinstance(value, str):
            parts = value.split(".")
            if len(parts) < 2:
                raise ValueError("Operation path must include its Python module")
            for part in parts:
                if not part.isidentifier() or keyword.iskeyword(part):
                    raise ValueError(
                        f"Invalid operation path {value!r}. "
                        "Use a dotted Python module and function name."
                    )
        return value

    def find_references(self) -> list[Reference]:
        """Find dependencies in the operation target and configured arguments."""
        return find_references(self.call) + find_references(self.kwargs)
