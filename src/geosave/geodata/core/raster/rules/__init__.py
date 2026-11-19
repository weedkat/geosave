"""Validation-only callables for native raster objects.

Rules inspect metadata and raise RasterValidationError. They do not change
representations, compute pixels, or select a conversion strategy.
"""

from collections.abc import Callable


type Rule[DataT] = Callable[[DataT], None]


__all__ = ["Rule"]
