"""Published marks what one module offers the modules built after it."""

from __future__ import annotations

import pytest
import torch.nn as nn

from geosave.ml.torch.chain.published import (
    Published,
    fill_published_arguments,
    read_published,
)


class Encoder(nn.Module):
    pyramid_channels: Published[list[int]]
    pyramid_strides: Published[list[int]]
    unmarked: int

    def __init__(self) -> None:
        super().__init__()
        self.pyramid_channels = [768] * 4
        self.pyramid_strides = [16] * 4
        self.unmarked = 7


class Decoder(nn.Module):
    feature_channels: Published[int]

    def __init__(self) -> None:
        super().__init__()
        self.feature_channels = 256


def test_a_subclass_inherits_its_base_marks():
    class Bigger(Encoder):
        input_size: Published[int]

    assert read_published(Bigger) == {
        "pyramid_channels": list[int],
        "pyramid_strides": list[int],
        "input_size": int,
    }


def test_an_optional_parameter_is_still_wired():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: int | None = None) -> None:
            super().__init__()

    assert fill_published_arguments(Consumer, {"decoder": Decoder()}, {}) == {
        "feature_channels": 256
    }


def test_the_same_name_at_a_different_type_is_refused():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: list[int]) -> None:
            super().__init__()

    with pytest.raises(TypeError, match="publishes"):
        fill_published_arguments(Consumer, {"decoder": Decoder()}, {})


class Other(nn.Module):
    feature_channels: Published[int]

    def __init__(self) -> None:
        super().__init__()
        self.feature_channels = 512


def test_the_nearest_earlier_stage_publishing_a_name_supplies_it():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

    modules = {"a": Decoder(), "b": Other(), "c": nn.Identity()}

    assert fill_published_arguments(Consumer, modules, {}) == {"feature_channels": 512}


def test_an_explicit_argument_bypasses_publications():
    class Consumer(nn.Module):
        def __init__(self, feature_channels: int) -> None:
            super().__init__()

    assert fill_published_arguments(
        Consumer,
        {"a": Decoder(), "b": Decoder()},
        {"feature_channels": 64},
    ) == {"feature_channels": 64}
