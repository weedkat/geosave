"""Find workflow references and bind operation arguments to native values."""

from dataclasses import dataclass
import keyword


@dataclass(frozen=True)
class Reference:
    """Name an input or step result, optionally followed by attribute names."""

    path: str

    def __post_init__(self) -> None:
        for part in self.path.split("."):
            if not part.isidentifier() or keyword.iskeyword(part):
                raise ValueError(
                    f"Reference requires names and attributes: {self.path!r}"
                )

    @property
    def root(self) -> str:
        """Return the input or step name supplying this value."""
        return self.path.split(".", 1)[0]

    def read_value(self, values: dict[str, object]) -> object:
        """Look up the named value and read its attributes without copying it.

        Args:
            values: External inputs and completed step results, keyed by name.

        Returns:
            The referenced native object, such as a raster, tensor, or method.
            Its concrete type is determined by the input or producing operation.

        Raises:
            KeyError: The named input or step result is unavailable.
            AttributeError: A requested attribute is absent. Exceptions raised
                by attribute properties also propagate with reference context.
        """
        parts = self.path.split(".")
        value = values[parts[0]]
        for part in parts[1:]:
            try:
                value = getattr(value, part)
            except Exception as error:
                error.add_note(
                    f"Cannot read '!ref {self.path}': reading attribute {part!r} "
                    f"on a {type(value).__name__} failed. "
                    "Check the reference against the supplied input or step's returned value."
                )
                raise
        return value


def find_references(value: object) -> list[Reference]:
    """Collect references nested in dictionary values and list elements.

    Literal values contribute no references. This inspects declarations only;
    it does not read runtime attributes or execute operations.
    """
    if isinstance(value, Reference):
        return [value]

    found = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(find_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_references(item))
    return found


def bind_arguments(
    arguments: dict[str, object], values: dict[str, object]
) -> dict[str, object]:
    """Replace argument references with the inputs and step results they name.

    Args:
        arguments: Configured keyword arguments containing literals or references.
        values: External inputs and completed step results, keyed by name.

    Returns:
        Keyword arguments ready to pass to an operation. Dictionary and list
        containers are rebuilt; referenced objects retain their identity.

    Raises:
        KeyError: A reference names an unavailable input or step result.
        AttributeError: A referenced attribute is absent. Property exceptions
            propagate with the argument name attached.
    """
    bound = {}
    for name, value in arguments.items():
        try:
            bound[name] = _replace_references(value, values)
        except Exception as error:
            error.add_note(
                f"Cannot supply argument {name!r}. "
                "Check its configured references against the values returned "
                "by earlier steps or supplied as inputs."
            )
            raise
    return bound


def _replace_references(value: object, values: dict[str, object]) -> object:
    """Rebuild one argument's containers, replacing each Reference with its value."""
    if isinstance(value, Reference):
        return value.read_value(values)
    if isinstance(value, dict):
        return {key: _replace_references(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_references(item, values) for item in value]
    return value
