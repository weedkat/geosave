import pytest
import torch

from geosave.model.stages.decoder.unet import Unet


@pytest.mark.parametrize("stride", [14, 16])
def test_patch_features_produce_half_resolution_with_gradients(stride):
    decoder = Unet(
        pyramid_channels=[8] * 4,
        pyramid_strides=[stride] * 4,
        decoder_channels=(8, 8, 8, 8),
    )
    pyramid = [torch.randn(1, 8, 16, 16, requires_grad=True) for _ in range(4)]

    output = decoder(pyramid)

    assert output.shape == (1, 8, 16 * stride // 2, 16 * stride // 2)
    output.square().mean().backward()
    for feature in pyramid:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0
