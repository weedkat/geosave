"""Prepare: deferred until the notebook examples settle the flow."""

from __future__ import annotations

import geopandas as gpd
import pytest

from geosave_engine.model.spec import ModelSpec, Ref


def test_prepare_is_not_settled_yet(tmp_path):
    spec = ModelSpec.model_validate(
        {
            "schema_version": 2,
            "rasters": {"optical": {"variables": ["red"]}},
            "inputs": {"image": Ref("optical")},
        }
    )

    with pytest.raises(NotImplementedError, match="create_items"):
        spec.prepare(gpd.GeoDataFrame(), tmp_path / "prepared", on="label")
