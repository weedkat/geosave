from __future__ import annotations

import torch

from geosave.model.registry import build_model, list_models


def test_a_small_cnn_feeds_the_unet_decoder_and_a_segmentation_head() -> None:
    assert ("encoder", "small_cnn") in list_models().index

    model = build_model(
        {
            "encoder": {
                "name": "small_cnn",
                "init_args": {"in_channels": 4, "input_size": 64},
            },
            "decoder": {"name": "unet"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 3}},
        }
    )

    assert model(image=torch.rand(2, 4, 64, 64)).shape == (2, 3, 64, 64)


def test_the_pyramid_halves_four_times_from_stride_four() -> None:
    model = build_model({"encoder": {"name": "small_cnn", "init_args": {"width": 8}}})
    encoder = model.encoder

    pyramid = encoder.forward_pyramid(torch.rand(1, 3, 64, 64))

    assert [level.shape[1:] for level in pyramid] == [
        (8, 16, 16),
        (16, 8, 8),
        (32, 4, 4),
        (64, 2, 2),
    ]
    assert encoder.pyramid_strides == [4, 8, 16, 32]
