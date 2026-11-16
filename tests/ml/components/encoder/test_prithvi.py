from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


from geosave.ml.components.encoder.prithvi import BACKBONE_REGISTRY, Prithvi, PrithviTL


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


class PatchEmbed(nn.Module):
    patch_size = (1, 2, 2)


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.out_channels = [3] * 12
        self.patch_embed = PatchEmbed()
        self.received: torch.Tensor | None = None
        self.context: dict[str, torch.Tensor | None] = {}

    def forward(
        self, image: torch.Tensor, **context: torch.Tensor | None
    ) -> list[torch.Tensor]:
        return self.forward_features(image, **context)

    def forward_features(
        self, image: torch.Tensor, **context: torch.Tensor | None
    ) -> list[torch.Tensor]:
        self.received = image
        self.context = context
        return [image.flatten(2).transpose(1, 2) * self.scale]

    def prepare_features_for_image_model(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        return features


@pytest.mark.parametrize("encoder", [Prithvi, PrithviTL])
@pytest.mark.parametrize("temporal", [False, True])
def test_prithvi_reorders_geodata_temporal_pixels(
    monkeypatch: pytest.MonkeyPatch,
    encoder: type[Prithvi] | type[PrithviTL],
    temporal: bool,
) -> None:
    backbone = Backbone()

    def build(*args: object, **kwargs: object) -> Backbone:
        return backbone

    monkeypatch.setattr(BACKBONE_REGISTRY, "build", build)
    model = encoder(out_indices=[0], num_frames=2 if temporal else 1)
    shape = (2, 2, 3, 4, 4) if temporal else (2, 3, 4, 4)
    image = torch.arange(np.prod(shape).item(), dtype=torch.float32).reshape(shape)
    expected = image.transpose(1, 2) if temporal else image
    if isinstance(model, PrithviTL):
        frames = 2 if temporal else 1
        model.forward_pyramid(
            image,
            centroid=torch.tensor([[51.0, 11.0]] * 2),
            year=torch.full((2, frames), 2024),
            day_of_year=torch.full((2, frames), 366),
        )
        # The pretrained model counts days from zero.
        torch.testing.assert_close(
            backbone.context["temporal_coords"],
            torch.tensor([[[2024.0, 365.0]] * frames] * 2),
        )
        assert backbone.context["location_coords"].shape == (2, 2)
    else:
        model.forward_pyramid(image)
    torch.testing.assert_close(backbone.received, expected)
    model(image)
    torch.testing.assert_close(backbone.received, expected)
