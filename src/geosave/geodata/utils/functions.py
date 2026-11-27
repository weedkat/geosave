"""Select a function by its registered name or take it as given."""

from collections.abc import Mapping


def resolve_fn[F](function: F | str, functions: Mapping[str, F]) -> F:
    """Return `function` as given, or the one `functions` holds under its name.

    A configuration names a built-in function by string; code passes the
    function itself, so a project extends the built-ins without registering.

    Args:
        function: The function, or its name in `functions`.
        functions: Built-in functions by name.

    Returns:
        The selected function.

    Raises:
        KeyError: The name is not in `functions`; the message lists the names.

    Examples:
        >>> resolve_fn("centroid", CONTEXT_READERS)
        <function read_centroid ...>
    """
    if not isinstance(function, str):
        return function
    if function not in functions:
        raise KeyError(f"{function!r} is not one of {sorted(functions)}")
    return functions[function]
