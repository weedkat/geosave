import pytest
import torch

from geosave.model.stages.decoder.dpt import DPT


@pytest.mark.parametrize("stride", [14, 16])
def test_patch_features_produce_half_resolution_with_gradients(stride):
    decoder = DPT(
        pyramid_channels=[8] * 4,
        pyramid_strides=[stride] * 4,
        readout="ignore",
        intermediate_channels=(8, 8, 8, 8),
        fusion_channels=8,
    )
    pyramid = [torch.randn(1, 8, 16, 16, requires_grad=True) for _ in range(4)]

    output = decoder(pyramid, [None] * 4)

    assert output.shape == (1, 8, 16 * stride // 2, 16 * stride // 2)
    output.square().mean().backward()
    for feature in pyramid:
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0
