"""Ingestors, one per source collection."""

from .base import IngestOutput, Ingestor
from .sentinel1_rtc import Sentinel1RTCIngestor
from .sentinel2_l2a import Sentinel2L2AIngestor

__all__ = ["IngestOutput", "Ingestor", "Sentinel1RTCIngestor", "Sentinel2L2AIngestor"]
