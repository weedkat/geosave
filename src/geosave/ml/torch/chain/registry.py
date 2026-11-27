"""Name the module classes a chain configuration selects.

    {"encoder": {"name": "dinov3", "init_args": {"in_channels": 4}}}
        │ BuildSpec.complete    "dinov3" ──▶ DINOv3, every constructor argument written out
        ▼ BuildSpec.build       DINOv3(in_channels=4, ...)

Library modules register into `MODULES` where they are defined. A module's
name in the configuration is free; the registered `name` alone selects the
class.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from importlib import import_module
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from .published import fill_published_arguments


class BuildSpec(BaseModel):
    """Select a registered or an importable module class for construction.

    Attributes:
        name: Case-insensitive name in `MODULES`.
        class_path: Importable module and class name. Supply exactly one of
            `name` and `class_path`.
        init_args: Keyword arguments passed to the selected constructor.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1)
    class_path: str | None = None
    init_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        """Require exactly one construction selector.

        Returns:
            This specification, without importing or constructing the
            selected class.

        Raises:
            ValueError: Both selectors are given, or neither is.
        """
        if (self.name is None) == (self.class_path is None):
            raise ValueError("Specify exactly one of name or class_path")

        return self

    def find_class(self) -> type[nn.Module]:
        """Return the class this specification selects.

        Raises:
            KeyError: The name is not registered.
            AttributeError: The imported module has no such class.
        """
        if self.class_path is not None:
            module_name, _, attribute = self.class_path.rpartition(".")
            return getattr(import_module(module_name), attribute)

        # The validator guarantees a name when there is no class path.
        assert self.name is not None
        return MODULES[self.name]

    def complete(self, modules: Mapping[str, nn.Module]) -> BuildSpec:
        """Return this specification with every constructor argument written out.

        Args:
            modules: The chain's earlier modules, whose published values fill
                omitted arguments.

        Returns:
            A copy whose `init_args` hold the explicit arguments, the
            published ones, then the constructor's defaults.

        Raises:
            TypeError: An argument does not fit the constructor, or a
                published value does not fit its parameter.
        """
        cls = self.find_class()
        arguments = fill_published_arguments(cls, modules, self.init_args)
        bound = inspect.signature(cls).bind(**arguments)
        bound.apply_defaults()
        return self.model_copy(update={"init_args": dict(bound.arguments)})

    def build(self) -> nn.Module:
        """Construct the selected class with this specification's arguments."""
        return self.find_class()(**self.init_args)

    def release(self) -> dict[str, Any]:
        """Return this specification as a release stores it.

        Returns:
            The specification as JSON values, tuples as lists, with
            `pretrained` off, so loading a release reads its saved weights
            instead of upstream ones.
        """
        init_args = dict(self.init_args)
        if "pretrained" in init_args:
            init_args["pretrained"] = False
        return self.model_copy(update={"init_args": init_args}).model_dump(
            mode="json", exclude_none=True
        )


class Registry(Mapping[str, type[nn.Module]]):
    """Module classes by case-insensitive name.

    The registry reads as a mapping of lower-case names to classes.
    Registration happens when a module defining a class is imported, so the
    registry imports `imports` before its first lookup: a release names
    `dinov3` before anything has imported the module that defines it.

    Args:
        imports: Modules whose `register` decorators fill this registry with
            the library's own classes.

    Examples:
        >>> registry = Registry()
        >>> @registry.register("tiny")
        ... class Tiny(nn.Module): ...
        >>> registry["TINY"]
        <class 'Tiny'>
    """

    def __init__(self, imports: tuple[str, ...] = ()) -> None:
        self.imports = imports
        self._classes: dict[str, type[nn.Module]] = {}

    def __getitem__(self, name: str) -> type[nn.Module]:
        self._import_modules()
        try:
            return self._classes[name.casefold()]
        except KeyError:
            raise KeyError(
                f"Unknown name {name!r}; available: {list(self._classes)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        self._import_modules()
        return iter(self._classes)

    def __len__(self) -> int:
        self._import_modules()
        return len(self._classes)

    @property
    def families(self) -> tuple[str, ...]:
        """Return the family names in `imports` order, then `custom`."""
        names = []
        for package in self.imports:
            names.append(package.rpartition(".")[2])
        names.append("custom")
        return tuple(names)

    def find_family(self, cls: type[nn.Module]) -> str:
        """Return the family of a registered class.

        The family is the last part of the `imports` package that defines the
        class, such as `encoder` for `geosave.ml.torch.modules.encoder`. A
        class defined anywhere else is `custom`.
        """
        for package in self.imports:
            if cls.__module__ == package or cls.__module__.startswith(package + "."):
                return package.rpartition(".")[2]
        return "custom"

    def _import_modules(self) -> None:
        """Import the modules that register the library's own classes."""
        for module in self.imports:
            import_module(module)

    def register[C: type[nn.Module]](self, name: str) -> Callable[[C], C]:
        """Register a class under a name.

        Args:
            name: Case-insensitive name used in configurations and releases.

        Returns:
            Decorator that registers and returns the class unchanged.

        Raises:
            TypeError: Another class holds the name.
        """

        def decorator(cls: C) -> C:
            key = name.casefold()
            known = self._classes.get(key)

            # A notebook cell run again defines a new class under the same path.
            if known is not None and (known.__module__, known.__qualname__) != (
                cls.__module__,
                cls.__qualname__,
            ):
                raise TypeError(
                    f"{name!r} is already registered to {known.__qualname__}"
                )

            self._classes[key] = cls
            return cls

        return decorator


MODULES = Registry(
    imports=(
        "geosave.ml.torch.modules.encoder",
        "geosave.ml.torch.modules.decoder",
        "geosave.ml.torch.modules.head",
        "geosave.ml.torch.modules.monolith",
    )
)
