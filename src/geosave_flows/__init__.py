"""Prefect flows composing GeoSave: serializable configs, tasks, and flows.

Examples:
    >>> from geosave_flows import ingest
    >>> ingest(anchor, output="data/samples/s1", spec="configs/model_spec.yaml")
"""

from .ingest import ingest

__all__ = ["ingest"]
