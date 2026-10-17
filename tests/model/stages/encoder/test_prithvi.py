from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import default_collate

from geosave.geodata import Dataset, attrs, raster
from geosave.geodata.attrs import TimeSpec
from geosave.model.stages.encoder import prithvi
from geosave.model.stages.encoder.prithvi import BACKBONE_REGISTRY, Prithvi, PrithviTL


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


def _chip(data, column=0):
    return data.isel(y=slice(0, 4), x=slice(4 * column, 4 * column + 4))


def test_model_context_reads_the_raster_it_encodes(scene):
    context = prithvi.model_context(scene)
    torch.testing.assert_close(
        context["temporal_coords"], torch.tensor([[2024.0, 365.0], [2024.0, 0.0]])
    )
    torch.testing.assert_close(context["location_coords"], torch.tensor([51.0, 11.0]))


def test_prithvi_context_preserves_frame_order_and_calendar_values(
    scene: Dataset,
) -> None:
    times = prithvi.temporal_coords(scene)
    centre = prithvi.location_coords(scene)
    torch.testing.assert_close(times, torch.tensor([[2024.0, 365.0], [2024.0, 0.0]]))
    torch.testing.assert_close(centre, torch.tensor([51.0, 11.0]))
    assert times.dtype == centre.dtype == torch.float32


def test_context_encodes_bucket_label_not_midpoint(scene: Dataset) -> None:
    single = scene.isel(time=slice(1, 2))
    monthly = single.assign_coords(
        time=attrs.stamp_attrs(single.time, TimeSpec.from_resample("MS"))
    )
    torch.testing.assert_close(
        prithvi.temporal_coords(monthly), torch.tensor([[2024.0, 0.0]])
    )


@pytest.mark.parametrize("labels", [[], [np.datetime64("NaT", "ns")]])
def test_invalid_time_labels_raise(scene: Dataset, labels: list) -> None:
    bad = scene.isel(time=slice(0, len(labels))).assign_coords(
        time=np.asarray(labels, dtype="datetime64[ns]")
    )
    with pytest.raises(ValueError, match="time labels"):
        prithvi.temporal_coords(bad)


def test_missing_time_raises(scene: Dataset) -> None:
    with pytest.raises(ValueError, match="time"):
        prithvi.temporal_coords(scene.isel(time=0, drop=True))


def test_missing_grid_raises() -> None:
    with pytest.raises(ValueError, match="grid"):
        prithvi.location_coords(
            raster({"red": (("y", "x"), np.zeros((2, 2), dtype="float32"))})
        )


def test_prithvi_context_batches_per_chip(scene: Dataset) -> None:
    batch = default_collate(
        [
            {
                "temporal_coords": prithvi.temporal_coords(_chip(scene, column)),
                "location_coords": prithvi.location_coords(_chip(scene, column)),
            }
            for column in range(2)
        ]
    )
    assert batch["temporal_coords"].shape == (2, 2, 2)
    torch.testing.assert_close(
        batch["location_coords"], torch.tensor([[51.5, 10.5], [51.5, 11.5]])
    )


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
        times = torch.zeros(2, 2 if temporal else 1, 2)
        locations = torch.zeros(2, 2)
        model.forward_pyramid(image, temporal_coords=times, location_coords=locations)
        assert backbone.context["temporal_coords"] is times
        assert backbone.context["location_coords"] is locations
    else:
        model.forward_pyramid(image)
    torch.testing.assert_close(backbone.received, expected)
    model(image)
    torch.testing.assert_close(backbone.received, expected)
