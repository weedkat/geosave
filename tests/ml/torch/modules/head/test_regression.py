from __future__ import annotations

import torch

from geosave.ml.torch.chain import ModelChain
from geosave.ml.torch.modules.head.regression import RegressionHead
from geosave.ml.torch.chain.catalog import list_models


def test_the_variables_fix_the_channel_count() -> None:
    head = RegressionHead(
        variables=["biomass", "height"], feature_channels=4, units="t/ha"
    )

    values = head(torch.rand(2, 4, 8, 8))

    assert values.shape == (2, 2, 8, 8)
    assert head.variables == ["biomass", "height"]
    assert head.units == "t/ha"


def test_the_head_ends_a_chain_under_its_registered_name() -> None:
    assert "regression" in list_models().loc["head"].index

    chain = ModelChain(head=RegressionHead(variables=["biomass"], feature_channels=4))

    assert chain(feature_map=torch.rand(1, 4, 8, 8)).shape == (1, 1, 8, 8)
