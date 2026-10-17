"""Declare the named inputs and outputs of ordinary module methods."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import inspect
import textwrap
import types
from typing import Any, ParamSpec, TypeVar, Union, get_args, get_origin, get_type_hints

from torch import nn

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _runtime_type(expected: type) -> type | tuple[type, ...]:
    """Resolve an annotation to the outer types accepted by isinstance."""
    origin = get_origin(expected)
    members = get_args(expected) if origin in (Union, types.UnionType) else (expected,)
    runtime: tuple[type, ...] = tuple(
        get_origin(member) or member for member in members
    )
    if any(member is Any or not isinstance(member, type) for member in runtime):
        raise TypeError(f"{expected} has no runtime outer type")
    for member in runtime:
        try:
            isinstance(None, member)
        except TypeError as error:
            raise TypeError(f"{expected} has no runtime outer type") from error
    return runtime[0] if len(runtime) == 1 else runtime


def type_name(kind: object) -> str:
    """Return a concise display name for an annotation.

    Args:
        kind: Type, parameterized annotation, or object to display.

    Returns:
        Type name or string representation, such as `Tensor` or `list[int]`.
    """
    return str(kind) if get_origin(kind) else getattr(kind, "__name__", str(kind))


def _validate_value(value: object, expected: type, where: str) -> None:
    """Check the outer runtime type of one annotated value.

    Args:
        value: Input or output value.
        expected: Resolved annotation; container elements are not inspected.
        where: Method and value name for an error message.

    Raises:
        TypeError: The value does not match its annotation.
    """
    if not isinstance(value, _runtime_type(expected)):
        raise TypeError(
            f"{where} expected {type_name(expected)}, got {type(value).__name__}"
        )


@dataclass(frozen=True)
class Step:
    """Describe one method's contract for named data flow.

    Args:
        name: Module method name.
        inputs: Required input names and annotations.
        optional_inputs: Inputs whose method parameters have defaults.
        outputs: Output names and annotations; empty for a terminal head.
        returns_tuple: Whether to unpack the method result into multiple outputs.
        result_type: Annotated type of a terminal result; None for named outputs.
    """

    name: str
    inputs: dict[str, type] = field(compare=False)
    optional_inputs: dict[str, type] = field(compare=False)
    outputs: dict[str, type] = field(compare=False)
    returns_tuple: bool = field(default=False, compare=False)
    result_type: type | None = field(default=None, compare=False)

    @property
    def head(self) -> bool:
        """Return whether the method produces a terminal result."""
        return not self.outputs

    def invoke(self, module: nn.Module, context: Mapping[str, object]) -> object:
        """Call a module method with its declared inputs and validate its result.

        Args:
            module: Module containing the declared method.
            context: Available values by name. Missing optional values use defaults.

        Returns:
            Named outputs, or the unchanged native result of a terminal head.

        Raises:
            KeyError: A required input is absent or None.
            TypeError: A value has the wrong type or output arity.
        """
        where = f"{type(module).__name__}.{self.name}"
        kwargs: dict[str, object] = {}
        for name, expected in (self.inputs | self.optional_inputs).items():
            value = context.get(name)
            if value is None:
                if name in self.inputs:
                    raise KeyError(f"{where}: missing input {name!r}")
                continue
            _validate_value(value, expected, f"{where}: input {name!r}")
            kwargs[name] = value

        # Calling the module preserves PyTorch hooks when its step is forward.
        method = module if self.name == "forward" else getattr(module, self.name)
        result = method(**kwargs)
        if self.head:
            assert self.result_type is not None
            _validate_value(result, self.result_type, f"{where}: result")
            return result
        if self.returns_tuple:
            if not isinstance(result, tuple) or len(result) != len(self.outputs):
                raise TypeError(f"{where}: expected a {len(self.outputs)}-tuple")
            values = result
        else:
            values = (result,)
        outputs = dict(zip(self.outputs, values))
        for name, expected in self.outputs.items():
            _validate_value(outputs[name], expected, f"{where}: output {name!r}")
        return outputs


def _infer_output_names(method: Callable[..., object]) -> tuple[str, ...]:
    """Infer names from one return statement in the method's own scope.

    Args:
        method: Function returning a local name or a tuple of local names.

    Returns:
        Returned variable names in order.

    Raises:
        TypeError: Source is unavailable or the return names are ambiguous.
    """
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    except (OSError, TypeError) as error:
        raise TypeError(
            f"{method.__qualname__}: no source is available; set outputs=(...)"
        ) from error
    pending = list(ast.iter_child_nodes(node))
    returns: list[ast.Return] = []
    while pending:
        child = pending.pop()
        if isinstance(child, ast.Return):
            returns.append(child)
        elif not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            pending.extend(ast.iter_child_nodes(child))
    if len(returns) != 1:
        raise TypeError(
            f"{method.__qualname__}: expected exactly one return statement; set outputs=(...)"
        )
    value = returns[0].value
    values = value.elts if isinstance(value, ast.Tuple) else [value]
    if not all(isinstance(item, ast.Name) for item in values):
        raise TypeError(
            f"{method.__qualname__}: return must be `return name` or a tuple of names; "
            "set outputs=(...) for expressions"
        )
    return tuple(item.id for item in values if isinstance(item, ast.Name))


def chain_step(
    head: bool = False, *, outputs: tuple[str, ...] | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Declare a chain step without changing the method's calling convention.

    Inputs and types come from annotations. Output names come from returned
    variables unless overridden. Defaulted parameters are optional inputs.

    Args:
        head: Return a terminal result instead of adding named chain outputs.
        outputs: Unique output names in return order. Bypasses source inspection;
            return annotations still determine types. Incompatible with head=True.

    Returns:
        Decorator that attaches a Step and returns the original method.

    Raises:
        TypeError: Annotations, parameter kinds, or output names are invalid;
            inference is ambiguous; or an input/output pair forms a self-cycle.

    Examples:
        >>> @chain_step(outputs=("pyramid", "prefix_tokens"))
        ... def forward_pyramid(self, image: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        ...     return self.backbone(image)
    """
    if head and outputs is not None:
        raise TypeError("head=True cannot declare outputs")

    def decorate(method: Callable[_P, _R]) -> Callable[_P, _R]:
        if head:
            setattr(method, "_chain_step", terminal_step(method))
            return method
        hints = get_type_hints(method)
        inputs, optional = _declared_inputs(method, hints)
        provided, returns_tuple = _declared_outputs(method, hints, outputs)
        if set(inputs.items()) & set(provided.items()):
            raise TypeError(
                f"{method.__qualname__}: self-cycle; use different output names"
            )
        step = Step(method.__name__, inputs, optional, provided, returns_tuple)
        setattr(method, "_chain_step", step)
        return method

    return decorate


def terminal_step(method: Callable[..., object]) -> Step:
    """Declare a method that returns a terminal result from its annotated inputs.

    Args:
        method: Function whose parameters name its inputs.

    Returns:
        A head step for `method`.

    Raises:
        TypeError: A parameter or the return lacks a usable annotation.
    """
    hints = get_type_hints(method)
    inputs, optional = _declared_inputs(method, hints)
    return Step(
        method.__name__,
        inputs,
        optional,
        outputs={},
        result_type=_terminal_type(method, hints),
    )


def _declared_inputs(
    method: Callable[..., object], hints: dict[str, Any]
) -> tuple[dict[str, type], dict[str, type]]:
    """Split annotated parameters into required and defaulted inputs.

    Raises:
        TypeError: A parameter has no annotation or cannot be passed by keyword.
    """
    inputs: dict[str, type] = {}
    optional: dict[str, type] = {}
    for name, parameter in inspect.signature(method).parameters.items():
        if name == "self":
            continue
        if name not in hints:
            raise TypeError(f"{method.__qualname__}: missing type hint(s) for {name!r}")
        if parameter.kind not in (
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"{method.__qualname__}: {name!r} must accept a keyword argument"
            )
        target = inputs if parameter.default is parameter.empty else optional
        target[name] = hints[name]
    return inputs, optional


def _terminal_type(method: Callable[..., object], hints: dict[str, Any]) -> type:
    """Return a head's return annotation, which must have a runtime outer type.

    Raises:
        TypeError: The return annotation is absent, None, or not checkable.
    """
    result_type = hints.get("return")
    if result_type is None or result_type is type(None):
        raise TypeError(
            f"{method.__qualname__}: terminal result needs a runtime return annotation"
        )
    try:
        _runtime_type(result_type)
    except TypeError as error:
        raise TypeError(f"{method.__qualname__}: terminal result {error}") from error
    return result_type


def _declared_outputs(
    method: Callable[..., object],
    hints: dict[str, Any],
    outputs: tuple[str, ...] | None,
) -> tuple[dict[str, type], bool]:
    """Pair output names with return annotation types.

    Args:
        method: Decorated method.
        hints: Its resolved annotations.
        outputs: Explicit names, or None to infer them from the return statement.

    Returns:
        Output names mapped to types, and whether the result is a tuple to unpack.

    Raises:
        TypeError: The return annotation is not fixed-arity, or the names are
            empty, repeated, or a different count from the annotation.
    """
    result_type = hints.get("return")
    if result_type is None or result_type is type(None) or result_type is tuple:
        raise TypeError(
            f"{method.__qualname__}: return types must be concrete and fixed-arity"
        )
    returns_tuple = get_origin(result_type) is tuple
    return_types: tuple[type, ...] = (
        get_args(result_type) if returns_tuple else (result_type,)
    )
    if not return_types or Ellipsis in return_types:
        raise TypeError(f"{method.__qualname__}: return types must be fixed-arity")
    names = _infer_output_names(method) if outputs is None else outputs
    if (
        not isinstance(names, tuple)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise TypeError("outputs must be a non-empty tuple of unique names")
    if len(names) != len(return_types):
        raise TypeError("Output names and return annotation arity must match")
    return dict(zip(names, return_types)), returns_tuple
