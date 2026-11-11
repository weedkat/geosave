"""Geometry checks shared by spatial tables and individual records."""

from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from geosave.geodata.errors import VectorValidationError


def validate_geometry(geometry: BaseGeometry | None, *, area: bool = False) -> None:
    """Require a present, nonempty, valid geometry without repairing it."""
    if geometry is None:
        raise VectorValidationError(
            "Geometry is missing. Supply a geometry or remove the row."
        )
    if geometry.is_empty:
        raise VectorValidationError(
            "Geometry is empty. Supply a nonempty geometry or remove the row."
        )
    if not geometry.is_valid:
        raise VectorValidationError(
            f"Geometry is invalid: {explain_validity(geometry)}. "
            "Review and repair the geometry before this operation."
        )
    if area and geometry.area == 0:
        raise VectorValidationError(
            "Geometry has no area; sampling requires polygon footprints."
        )
