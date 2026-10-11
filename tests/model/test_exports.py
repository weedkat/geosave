"""Names `geosave_engine.model` offers in one import."""

from __future__ import annotations

import pytest


def test_model_exports_the_chain_and_its_builder() -> None:
    from geosave_engine.model import ModelChain, Published, build_model, chain_step
    from geosave_engine.model import chain, registry

    assert ModelChain is chain.ModelChain
    assert Published is chain.Published
    assert chain_step is chain.chain_step
    assert build_model is registry.build_model


def test_model_refuses_a_name_it_does_not_export() -> None:
    import geosave_engine.model as model

    with pytest.raises(AttributeError, match="no_such_name"):
        model.no_such_name
