"""Read the fields a `str.format` template names."""

from __future__ import annotations

from string import Formatter


def fields_named(template: str) -> dict[str, str | None]:
    """Map each field a template names to the format spec it carries.

    Args:
        template: Python format string, such as `"{time:%Y%m%d}_{band}.tif"`.

    Returns:
        Field names mapped to format specs, with an empty string for an
        unspecified spec. A repeated field takes its last spec.

    Raises:
        ValueError: The template has malformed format-string syntax.

    Examples:
        >>> fields_named("{time:%Y%m%d}_{band}.tif")
        {'time': '%Y%m%d', 'band': ''}
    """
    return {name: spec for _, name, spec, _ in Formatter().parse(template) if name}
