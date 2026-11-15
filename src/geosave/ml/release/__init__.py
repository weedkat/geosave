"""Release native model chains and their processors as Transformers models.

Importing this package registers `ChainModel` with `AutoModel`.
"""

from .huggingface import ChainConfig, ChainModel, load_processor

__all__ = ["ChainConfig", "ChainModel", "load_processor"]
