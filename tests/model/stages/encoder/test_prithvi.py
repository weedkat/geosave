from pathlib import Path

import torch

from geosave.model.stages.encoder.prithvi import PrithviTL


def test_named_bands_keep_their_pretrained_patch_embedding(tmp_path: Path) -> None:
    pretrained = PrithviTL(model_name="prithvi_eo_v2_tiny_tl", input_size=32)
    checkpoint = tmp_path / "prithvi.pt"
    torch.save(pretrained.model.state_dict(), checkpoint)

    encoder = PrithviTL(
        model_name="prithvi_eo_v2_tiny_tl",
        pretrained=True,
        ckpt_path=str(checkpoint),
        bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "elevation"],
        input_size=32,
    )

    # Patch-embedding columns follow the pretrained order BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2.
    weight = encoder.model.patch_embed.proj.weight
    source = pretrained.model.patch_embed.proj.weight
    assert weight.shape[1] == 5
    assert torch.equal(weight[:, :4], source[:, :4])
    assert not torch.equal(weight[:, 4], source[:, 4])


def test_default_bands_are_the_pretrained_six() -> None:
    encoder = PrithviTL(model_name="prithvi_eo_v2_tiny_tl", input_size=32)

    assert encoder.model.patch_embed.proj.weight.shape[1] == 6
