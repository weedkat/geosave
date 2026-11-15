"""Select and construct objects by registered name or class path.

    BuildSpec(name="adamw", init_args={"lr": 1e-3})
        │ Registry.resolve   name or class path ──▶ constructor
        │ Registry.get_spec  every constructor argument written out
        ▼ Registry.build     constructor(*args, **init_args)
    object

Each configurable kind (encoders, predictors, optimizers, ...) has one
`Registry`; its owner declares it beside the classes it holds.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from importlib import import_module
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BuildSpec(BaseModel):
    """Select a registered factory or importable class for construction.

    Attributes:
        name: Case-insensitive factory name in the caller's registry.
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
        """Require exactly one valid construction selector.

        Returns:
            This specification after selector validation, without importing
            or constructing the selected factory.

        Raises:
            ValueError: Both selectors are given, neither is given, or a
                class path does not include a module name.
        """
        if (self.name is None) == (self.class_path is None):
            raise ValueError("Specify exactly one of name or class_path")
        if self.class_path is not None and "." not in self.class_path:
            raise ValueError(
                "class_path must include an importable module and class name"
            )
        return self


class Registry[T](Mapping[str, Callable[..., T]]):
    """Constructors of one base type, looked up by case-insensitive name.

    The registry reads as a mapping of lower-case names to constructors.
    Registration happens when a module defining a constructor is imported,
    so the registry imports `imports` before its first lookup: a release
    names `chips` before anything has imported the module that defines it.

    Args:
        base: Type every constructor builds. A registered class, or a class
            selected by class path, must subclass it.
        imports: Modules whose `register` decorators fill this registry with
            the library's own constructors.

    Examples:
        >>> ENCODERS = Registry(nn.Module, imports=("geosave.ml.components.encoder",))
        >>> @ENCODERS.register("tiny")
        ... class Tiny(nn.Module): ...
        >>> ENCODERS.build(BuildSpec(name="TINY"))
        Tiny()
    """

    def __init__(self, base: type[T], imports: tuple[str, ...] = ()) -> None:
        self.base = base
        self.imports = imports
        self._factories: dict[str, Callable[..., T]] = {}

    def __getitem__(self, name: str) -> Callable[..., T]:
        self.import_modules()
        try:
            return self._factories[name.casefold()]
        except KeyError:
            raise KeyError(
                f"Unknown name {name!r}; available: {list(self._factories)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        self.import_modules()
        return iter(self._factories)

    def __len__(self) -> int:
        self.import_modules()
        return len(self._factories)

    def import_modules(self) -> None:
        """Import the modules that register the library's own constructors."""
        for module in self.imports:
            import_module(module)

    def register[F: Callable[..., Any]](self, name: str) -> Callable[[F], F]:
        """Register a class or factory function under a name.

        Args:
            name: Case-insensitive name used in configurations and releases.

        Returns:
            Decorator that registers and returns the constructor unchanged.

        Raises:
            TypeError: A registered class does not subclass `base`.
            ValueError: The name already identifies another constructor. One
                of the same module and qualified name, as a notebook cell run
                again defines, replaces the one registered.
        """

        def decorator(factory: F) -> F:
            self.check_base(factory)

            key = name.casefold()
            existing = self._factories.get(key)
            if existing is not None and existing is not factory:
                # A notebook cell run again defines a new class under the same path.
                same_path = (existing.__module__, existing.__qualname__) == (
                    factory.__module__,
                    factory.__qualname__,
                )
                if not same_path:
                    raise ValueError(f"{name!r} already identifies another factory")

            self._factories[key] = factory
            return factory

        return decorator

    def check_base(self, factory: Callable[..., Any]) -> None:
        """Reject a class that does not subclass `base`; functions pass unchecked.

        Raises:
            TypeError: The class does not subclass `base`.
        """
        if isinstance(factory, type) and not issubclass(factory, self.base):
            raise TypeError(f"{factory.__qualname__} is not a {self.base.__name__}")

    def resolve(self, spec: BuildSpec) -> Callable[..., T]:
        """Return the constructor a specification selects.

        A class path reaches any importable subclass of `base`, registered or
        not; a name looks up a registered constructor.

        Raises:
            KeyError: The registered name is unknown.
            AttributeError: The imported module has no such class.
            TypeError: An imported class does not subclass `base`.
        """
        if spec.class_path is not None:
            return self.import_class(spec.class_path)

        # BuildSpec's validator guarantees a name when there is no class path.
        assert spec.name is not None
        return self[spec.name]

    def import_class(self, class_path: str) -> type[T]:
        """Import a class by its dotted path and check it subclasses `base`.

        Raises:
            AttributeError: The imported module has no such class.
            TypeError: The path names no class, or one outside `base`.
        """
        module_name, _, attribute = class_path.rpartition(".")
        factory = getattr(import_module(module_name), attribute)
        if not isinstance(factory, type):
            raise TypeError(f"{class_path!r} must name a class")
        self.check_base(factory)
        return factory

    def get_spec[S: BuildSpec](self, spec: S) -> S:
        """Return the spec a release stores: every constructor argument written out.

        A configuration gives only some arguments. Writing out the defaults
        too makes a release rebuild the same object, even after a later
        version changes those defaults.

        Args:
            spec: Selector and the arguments a configuration gives.

        Returns:
            Copy of `spec` whose `init_args` hold every argument, defaults
            included.

        Raises:
            TypeError: An argument does not fit the constructor.

        Examples:
            >>> PREDICTORS.get_spec(BuildSpec(name="chips", init_args={"size": 256}))
            BuildSpec(name='chips', class_path=None, init_args={'size': 256,
            'stride': None, 'mode': 'reflect', 'window': 'hann', 'batch_size': 8})
        """
        init_args = bind_init_args(self.resolve(spec), spec.init_args)
        return spec.model_copy(update={"init_args": init_args})

    def build(self, spec: BuildSpec, *args: object) -> T:
        """Construct the selected object.

        Args:
            spec: Selector and keyword arguments.
            *args: Live objects a configuration cannot hold, passed first:
                an optimizer's parameter groups, a scheduler's optimizer.

        Returns:
            The constructed object.
        """
        return self.resolve(spec)(*args, **spec.init_args)


def bind_init_args(
    factory: Callable[..., object], init_args: Mapping[str, object]
) -> dict[str, object]:
    """Bind keyword arguments to a constructor and fill in its defaults.

    Args:
        factory: Class or function to be called.
        init_args: Keyword arguments for it.

    Returns:
        Independent copy of every keyword argument the call receives;
        arguments caught by `**kwargs` appear under their own names.

    Raises:
        TypeError: An argument does not fit the constructor.
    """
    signature = inspect.signature(factory)
    bound = signature.bind(**init_args)
    bound.apply_defaults()

    arguments: dict[str, object] = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            arguments[name] = value
    return deepcopy(arguments)
