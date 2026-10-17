"""Inert references and complete model-owned call declarations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from importlib import import_module
import math
import re

from pydantic import ConfigDict, Field, field_validator

from .base import SpecModel

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
# An importable call: dotted names only.
_CALL_PATH = re.compile(rf"{_NAME}(?:\.{_NAME})*")
# A reference step: `.attribute`, or `["key"]` for a name no attribute can
# spell, such as the STAC property `view:sun_azimuth`.
_STEP = re.compile(rf'\.({_NAME})|\["([^"\]]+)"\]')
_REF_PATH = re.compile(rf'({_NAME})(?:\.{_NAME}|\["[^"\]]+"\])*')


class Ref:
    """Name a supplied value, one of its attributes, or one of its keys.

    Inside a YAML flow mapping, quote a path holding a key:
    `{azimuth: !ref 'scene["view:sun_azimuth"]'}`.

    Args:
        path: Value name followed by `.attribute` and `["key"]` steps.

    Examples:
        >>> Ref('scene["view:sun_azimuth"]').resolve({"scene": {"view:sun_azimuth": 151.2}})
        151.2
    """

    __slots__ = ("_path",)

    def __init__(self, path: str) -> None:
        if not isinstance(path, str) or not _REF_PATH.fullmatch(path):
            raise ValueError(f"Invalid reference path: {path!r}")
        self._path = path

    @property
    def path(self) -> str:
        """Return the referenced Python path."""
        return self._path

    @property
    def root(self) -> str:
        """Return the name required in the supplied inputs."""
        match = _REF_PATH.match(self.path)
        assert match is not None
        return match.group(1)

    def resolve(self, inputs: Mapping[str, object]) -> object:
        """Retrieve the value, following attributes and keys without calling methods.

        Args:
            inputs: Runtime roots by name; this reference selects its root
                and then follows the declared attribute and key steps.

        Returns:
            The referenced object unchanged. Attribute lookups retain their
            native behavior, including properties that can compute values.

        Raises:
            KeyError: A value or key is absent; the note names this path.
            AttributeError: An attribute is absent; the note names this path.
            TypeError: A key step meets a value without keys; the note names
                this path.
        """
        result: Any
        try:
            result = inputs[self.root]
            for attribute, key in _STEP.findall(self.path, len(self.root)):
                result = getattr(result, attribute) if attribute else result[key]
        except (KeyError, AttributeError, TypeError) as error:
            error.add_note(f"While resolving !ref {self.path}")
            raise
        return result

    def __repr__(self) -> str:
        """Show the reference as constructor-like text.

        Returns:
            `Ref('path')` text containing the stored reference path.
        """
        return f"Ref({self.path!r})"

    def __eq__(self, other: object) -> bool:
        """Compare references by their stored paths without resolving values.

        Args:
            other: Reference or other object to compare with this one.

        Returns:
            True for a `Ref` with the same path; False for other objects.
        """
        return isinstance(other, Ref) and self.path == other.path

    def __hash__(self) -> int:
        """Hash the stored path for dictionary and set membership.

        Returns:
            Integer hash of the reference path.
        """
        return hash(self.path)


type CallValue = (
    None | bool | int | float | str | Ref | list[CallValue] | dict[str, CallValue]
)


class CallSpec(SpecModel):
    """Declare one inert call and its literal or referenced keyword arguments.

    Attributes:
        call: Importable module-level callable path, or a reference to a
            callable on the supplied runtime values.
        kwargs: Named arguments containing finite YAML literals, references,
            or nested lists/mappings of those values. Empty by default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: str | Ref
    kwargs: dict[str, CallValue] = Field(default_factory=dict)

    @field_validator("call")
    @classmethod
    def validate_call(cls, value: str | Ref) -> str | Ref:
        """Require a reference or a module-level call path without importing it.

        Args:
            value: Callable path or runtime reference to validate.

        Returns:
            The original path or reference.

        Raises:
            ValueError: A string is not a dotted callable path.
        """
        if isinstance(value, Ref):
            return value
        if not _CALL_PATH.fullmatch(value):
            raise ValueError(f"Invalid call path: {value!r}")
        if "." not in value:
            raise ValueError("An imported call needs a module and callable name")
        return value

    @field_validator("kwargs", mode="before")
    @classmethod
    def validate_kwargs(cls, value: object) -> object:
        """Accept finite YAML literals and inert references without cycles.

        Args:
            value: Keyword-argument structure to validate recursively.

        Returns:
            Validated literals/references, copying nested lists and mappings.

        Raises:
            ValueError: A value is unsupported or non-finite, a mapping key
                is not text, or containers contain a cycle.
        """
        return cls._validate_value(value)

    @classmethod
    def _validate_value(cls, value: object, active: set[int] | None = None) -> object:
        if value is None or isinstance(value, (Ref, str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        if not isinstance(value, (dict, list)):
            raise ValueError(
                f"Expected a YAML literal or Ref, got {type(value).__name__}"
            )
        active = set() if active is None else active
        if id(value) in active:
            raise ValueError("Cyclic configuration values are not supported")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("Configuration mapping keys must be strings")
                return {
                    key: cls._validate_value(item, active)
                    for key, item in value.items()
                }
            return [cls._validate_value(item, active) for item in value]
        finally:
            active.remove(id(value))

    @classmethod
    def _find_references(cls, value: object) -> tuple[Ref, ...]:
        if isinstance(value, Ref):
            return (value,)
        if isinstance(value, dict):
            return tuple(
                reference
                for item in value.values()
                for reference in cls._find_references(item)
            )
        if isinstance(value, list):
            return tuple(
                reference for item in value for reference in cls._find_references(item)
            )
        return ()

    @property
    def references(self) -> tuple[Ref, ...]:
        """Return every call and argument reference in encounter order."""
        return self._find_references([self.call, self.kwargs])

    @property
    def inputs(self) -> frozenset[str]:
        """Return the runtime roots required by this declaration."""
        return frozenset(reference.root for reference in self.references)

    def select_inputs(self, state: Mapping[str, object]) -> dict[str, object]:
        """Select only the runtime roots this call references.

        Args:
            state: Available runtime values by root name.

        Returns:
            A new mapping of referenced names to their existing objects,
            without resolving nested attributes or reading pixels.

        Raises:
            KeyError: A referenced root is absent from `state`.
        """
        if missing := self.inputs - state.keys():
            raise KeyError(f"Missing call inputs: {sorted(missing)}")
        return {reference.root: state[reference.root] for reference in self.references}

    @classmethod
    def _resolve_value(cls, value: CallValue, inputs: Mapping[str, object]) -> object:
        if isinstance(value, Ref):
            return value.resolve(inputs)
        if isinstance(value, dict):
            return {
                key: cls._resolve_value(item, inputs) for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._resolve_value(item, inputs) for item in value]
        return value

    def _resolve_target(self, inputs: Mapping[str, object]) -> Callable[..., object]:
        if isinstance(self.call, Ref):
            target = self.call.resolve(inputs)
        else:
            module, name = self.call.rsplit(".", 1)
            target = getattr(import_module(module), name)
        if not callable(target):
            raise TypeError(f"Expected a callable, got {type(target).__name__}")
        return target

    def invoke(self, inputs: Mapping[str, object], /) -> object:
        """Resolve the target and arguments, then invoke this declaration once.

        Args:
            inputs: Runtime roots used by the callable and argument references.

        Returns:
            The callable's result unchanged. Its own contract determines
            whether execution reads pixels, computes, or has side effects.

        Raises:
            TypeError: The selected target is not callable. Exceptions from
                resolution or execution propagate with this call's context.
        """
        try:
            target = self._resolve_target(inputs)
            kwargs = {
                name: self._resolve_value(value, inputs)
                for name, value in self.kwargs.items()
            }
            return target(**kwargs)
        except Exception as error:
            error.add_note(f"While invoking {self.call!r}")
            raise
