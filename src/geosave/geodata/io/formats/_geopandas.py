"""Prepare shared GeoPandas file-driver options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def unpack_engine_options(open_options: Mapping[str, Any]) -> dict[str, Any]:
    """Join typed read options with `engine_options`, refusing any given twice.

    Args:
        open_options: Read options, possibly holding `engine_options`.

    Returns:
        One mapping of keywords for `gpd.read_file`.

    Raises:
        TypeError: A key is both a typed option and a driver option.
        ValueError: A driver option selects an engine or disables geometry.
    """
    typed = dict(open_options)
    engine_options = dict(typed.pop("engine_options", {}))
    if "engine" in engine_options:
        raise ValueError("engine_options must not contain engine; use engine=")
    if "ignore_geometry" in engine_options:
        raise ValueError("engine_options must not contain ignore_geometry")
    if repeated := sorted(typed.keys() & engine_options.keys()):
        raise TypeError(
            f"{repeated} given both directly and in engine_options; pass each once"
        )
    return {**typed, **engine_options}
