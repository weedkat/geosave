from __future__ import annotations

import torch

from geosave.model.registry import build_model


def test_the_monolith_runs_alone_in_a_chain_and_names_its_variable() -> None:
    chain = build_model(
        {"model": {"name": "ibm_granite_biomass", "init_args": {"pretrained": False}}}
    )

    assert chain.inputs == {"image": torch.Tensor}
    assert chain.get_submodule("model").variables == ["biomass"]
    assert chain.eval()(image=torch.rand(1, 6, 224, 224)).shape == (1, 1, 224, 224)
