from __future__ import annotations

import torch

from geosave.ml.torch import ModelChain


def test_the_monolith_runs_alone_in_a_chain_and_names_its_variable() -> None:
    chain = ModelChain.from_config(
        {"model": {"name": "ibm_granite_biomass", "init_args": {"pretrained": False}}}
    )

    assert [spec.name for spec in chain.inputs] == ["image"]
    assert chain.get_submodule("model").variables == ["biomass"]
    assert chain.eval()(image=torch.rand(1, 6, 224, 224)).shape == (1, 1, 224, 224)
