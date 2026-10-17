"""Warnings for data loss, assumptions, and incomplete results."""

__all__ = [
    "GeoSaveWarning",
    "UnreadMaskWarning",
    "AssumedFillWarning",
    "DroppedAttrsWarning",
    "DroppedInstantsWarning",
    "DroppedRowsWarning",
    "PartialCoverWarning",
]


class GeoSaveWarning(UserWarning):
    """Base category for GeoSave warnings.

    Escalate the family with ``warnings.simplefilter("error", GeoSaveWarning)``.
    """


class UnreadMaskWarning(GeoSaveWarning):
    """A band's absence mask is not read."""


class AssumedFillWarning(GeoSaveWarning):
    """A warped variable carried no fill value, so one was chosen."""


class DroppedAttrsWarning(GeoSaveWarning):
    """Conflicting or incomplete source metadata was dropped."""


class DroppedInstantsWarning(GeoSaveWarning):
    """Incomplete or trailing sampling positions were left out."""


class DroppedRowsWarning(GeoSaveWarning):
    """Malformed or ambiguous catalog rows were skipped."""


class PartialCoverWarning(GeoSaveWarning):
    """Items intersect a window without covering it, so they do not join it."""


class TimelessItemWarning(GeoSaveWarning):
    """An Item's files state no time, so it spans all of time."""


class MissingGridWarning(GeoSaveWarning):
    """An Item cannot define a spatial window because it states no grid."""
