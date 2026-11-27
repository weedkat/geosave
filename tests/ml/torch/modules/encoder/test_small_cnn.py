from __future__ import annotations

import torch

from geosave.ml.torch import ModelChain
from geosave.ml.torch.chain.catalog import list_models


def test_a_small_cnn_feeds_the_unet_decoder_and_a_segmentation_head() -> None:
    assert "small_cnn" in list_models().loc["encoder"].index

    model = ModelChain.from_config(
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
