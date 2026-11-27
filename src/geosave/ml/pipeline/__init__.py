"""The pipeline from a region to a published product, and its stages.

Each stage package holds one kind of stage, named by its identity:

- `ingestion`: one ingestor per source collection
- `preprocessing`: the preprocessor, from source rasters to model samples
- `prediction`: the predictor contract; predictors are framework code
- `postprocessing`: one postprocessor per task
- `publishing`: one publisher per product format
"""

from .ingestion import IngestOutput, Ingestor, Sentinel1RTCIngestor, Sentinel2L2AIngestor
from .pipeline import Pipeline
from .postprocessing import PostprocessOutput
from .prediction import PredictOutput
from .preprocessing import PreprocessOutput, Preprocessor

__all__ = [
    "IngestOutput",
    "Ingestor",
    "Pipeline",
    "PostprocessOutput",
    "PredictOutput",
    "PreprocessOutput",
    "Preprocessor",
    "Sentinel1RTCIngestor",
    "Sentinel2L2AIngestor",
]
