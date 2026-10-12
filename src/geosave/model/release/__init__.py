"""Release native model chains as Transformers models.

Importing this package registers `GeoSaveModel` with `AutoModel`.
"""

from .huggingface import GeoSaveConfig, GeoSaveModel

__all__ = ["GeoSaveConfig", "GeoSaveModel"]
