"""DINOv3 encoders: a feature pyramid, or one pooled embedding per image."""

import pytest
import torch

from geosave.ml.torch import ModelChain, list_models
from geosave.ml.torch.modules.encoder import DINOv3, DINOv3Embedding

MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"


@pytest.fixture
def image() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(2, 3, 32, 32)


def test_dinov3_returns_one_feature_map_and_prefix_tokens_per_level(image):
    encoder = DINOv3(model_name=MODEL_NAME, pretrained=False, input_size=32).eval()

    with torch.no_grad():
        pyramid, prefix_tokens = encoder(image)

    assert encoder.pyramid_channels == [384] * 4
    assert encoder.pyramid_strides == [16] * 4
    assert [tuple(level.shape) for level in pyramid] == [(2, 384, 2, 2)] * 4
    # One CLS and four register tokens.
    assert [tuple(tokens.shape) for tokens in prefix_tokens] == [(2, 5, 384)] * 4


def test_dinov3_embedding_is_timms_pooled_pre_logits_output(image):
    encoder = DINOv3Embedding(
        model_name=MODEL_NAME, pretrained=False, input_size=32
    ).eval()

    with torch.no_grad():
        embedding = encoder(image)
        expected = encoder.model(image)

    assert encoder.embedding_channels == 384
    assert embedding.shape == (2, 384)
    torch.testing.assert_close(embedding, expected)


def test_dinov3_embedding_runs_alone_as_a_chain():
    chain = ModelChain(
        encoder=DINOv3Embedding(model_name=MODEL_NAME, pretrained=False, input_size=32)
    )

    assert [parameter.name for parameter in chain.inputs] == ["image"]
    assert list_models().loc[("encoder", "dinov3_embedding"), "forward"] == (
        "forward(image) -> embedding"
    )
