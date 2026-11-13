"""Unavailable workflows fail before reading or executing user recipes."""

import pytest

from geosave.model.spec import ModelSpec


def test_model_spec_cannot_be_constructed():
    with pytest.raises(NotImplementedError, match="ModelSpec is being redesigned"):
        ModelSpec()


def test_model_spec_load_does_not_access_storage():
    with pytest.raises(NotImplementedError, match="ModelSpec is being redesigned"):
        ModelSpec.load("https://example.invalid/model_spec.yaml")
