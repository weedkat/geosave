"""Disabled model workflow entry point, pending redesign."""

from geosave.geodata.io.storage import Location, StorageOptions


class ModelSpec:
    """Unavailable while model workflow ownership and execution are redesigned."""

    def __init__(self) -> None:
        raise NotImplementedError("ModelSpec is being redesigned.")

    @classmethod
    def load(
        cls, path: Location, *, storage_options: StorageOptions | None = None
    ) -> "ModelSpec":
        """Raise before reading a recipe; ModelSpec is being redesigned."""
        raise NotImplementedError("ModelSpec is being redesigned.")
