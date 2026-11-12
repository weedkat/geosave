"""Exceptions for geodata conditions callers can handle separately."""

from __future__ import annotations


class RegionFetchError(RuntimeError):
    """A STAC search matched no items for a region."""


class CollectionNotFoundError(LookupError):
    """A STAC catalog does not publish the requested collection."""


class RasterValidationError(ValueError):
    """Explain invalid raster data without hiding unrelated programming errors.

    Args:
        message: Problem and suggested correction, understandable on its own.
        variable: DataArray or Dataset variable name, when known.
        path: Relative DataTree node path, when known.
        operation: Description of the operation that could not proceed.

    Examples:
        >>> error = RasterValidationError("No CRS. Assign the known CRS with raster.odc.assign_crs(...).", variable="red")
        >>> str(error)
        "Raster 'red': No CRS. Assign the known CRS with raster.odc.assign_crs(...)."
    """

    def __init__(
        self,
        message: str,
        *,
        variable: str | None = None,
        path: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.variable = variable
        self.path = path
        self.operation = operation

    def __str__(self) -> str:
        location = self.variable
        if self.path is not None and self.path != ".":
            location = self.path
            if self.variable is not None:
                location = f"{self.path}/{self.variable}"
        message = self.message
        if location is not None:
            message = f"Raster {location!r}: {message}"
        if self.operation is not None:
            message = f"{self.operation}. {message}"
        return message


class VectorValidationError(ValueError):
    """Explain invalid spatial records with row context and a suggested correction."""
