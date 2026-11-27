"""Deployment entry point for prediction over a geographic region."""

from prefect import flow

from geosave_flows.configs import RegionConfig


@flow
def predict_region(repo: str, region: RegionConfig, root: str) -> None:
    """Reserved until a released processor can stream geospatial products to storage."""
    raise NotImplementedError(
        "Region prediction needs pipeline loading and a bounded product writer"
    )
