"""Model chains, components, pipelines, releases and training support.

Public names load on first use, so `import geosave.ml` pays for neither
torch nor Lightning.

Examples:
    >>> from geosave.ml import ModelChain, build_model, chain_step, load_pipeline
    >>> from geosave.ml import RasterPipeline, RasterPredictor
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chain import ModelChain, chain_step
    from .components.build import build_model
    from .components.catalog import list_models
    from .components.published import Published
    from .factory import BuildSpec, Registry
    from .processing.change.pipeline import ChangePipeline
    from .processing.change.preprocessing.base import ChangePreprocessor
    from .processing.prompt.pipeline import PromptPipeline
    from .processing.prompt.postprocessing.base import PromptPostprocessor
    from .processing.prompt.prediction.base import PromptPredictor
    from .processing.raster.ingestion.base import RasterIngestor
    from .processing.raster.ingestion.sentinel2 import Sentinel2Ingestor
    from .processing.raster.pipeline import RasterPipeline
    from .processing.raster.postprocessing.base import RasterPostprocessor
    from .processing.raster.postprocessing.dense import (
        RegressionPostprocessor,
        SegmentationPostprocessor,
    )
    from .processing.raster.prediction.base import RasterPredictor
    from .processing.raster.prediction.dense import DensePredictor
    from .processing.raster.preprocessing.base import RasterPreprocessor
    from .processing.raster.preprocessing.dense import DensePreprocessor
    from .release import load_pipeline

_LAZY = {
    "BuildSpec": "factory",
    "ChangePipeline": "processing.change.pipeline",
    "ChangePreprocessor": "processing.change.preprocessing.base",
    "DensePredictor": "processing.raster.prediction.dense",
    "DensePreprocessor": "processing.raster.preprocessing.dense",
    "ModelChain": "chain",
    "PromptPipeline": "processing.prompt.pipeline",
    "PromptPostprocessor": "processing.prompt.postprocessing.base",
    "PromptPredictor": "processing.prompt.prediction.base",
    "Published": "components.published",
    "RasterIngestor": "processing.raster.ingestion.base",
    "RasterPipeline": "processing.raster.pipeline",
    "RasterPostprocessor": "processing.raster.postprocessing.base",
    "RasterPredictor": "processing.raster.prediction.base",
    "RasterPreprocessor": "processing.raster.preprocessing.base",
    "Registry": "factory",
    "RegressionPostprocessor": "processing.raster.postprocessing.dense",
    "SegmentationPostprocessor": "processing.raster.postprocessing.dense",
    "Sentinel2Ingestor": "processing.raster.ingestion.sentinel2",
    "build_model": "components.build",
    "chain_step": "chain",
    "list_models": "components.catalog",
    "load_pipeline": "release",
}

__all__ = [
    "BuildSpec",
    "ChangePipeline",
    "ChangePreprocessor",
    "DensePredictor",
    "DensePreprocessor",
    "ModelChain",
    "PromptPipeline",
    "PromptPostprocessor",
    "PromptPredictor",
    "Published",
    "RasterIngestor",
    "RasterPipeline",
    "RasterPostprocessor",
    "RasterPredictor",
    "RasterPreprocessor",
    "Registry",
    "RegressionPostprocessor",
    "SegmentationPostprocessor",
    "Sentinel2Ingestor",
    "build_model",
    "chain_step",
    "list_models",
    "load_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
