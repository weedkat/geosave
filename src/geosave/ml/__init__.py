"""Framework-free model support: the processing pipeline and its stages.

Each ML framework has its own subpackage. `geosave.ml.torch` holds PyTorch
model chains, modules, releases and Lightning training support.

Examples:
    >>> from geosave.ml import Pipeline, Sentinel2L2AIngestor
    >>> from geosave.ml.torch import ModelChain
"""

from .pipeline import (
    Ingestor,
    Pipeline,
    Preprocessor,
    Sentinel1RTCIngestor,
    Sentinel2L2AIngestor,
)

__all__ = [
    "Ingestor",
    "Pipeline",
    "Preprocessor",
    "Sentinel1RTCIngestor",
    "Sentinel2L2AIngestor",
]
