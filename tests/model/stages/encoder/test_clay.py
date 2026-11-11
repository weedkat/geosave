import pytest
import torch

from geosave.geodata import Dataset
from geosave.model.stages.encoder import clay


def test_local_checkpoint_needs_no_published_model_source(tmp_path, monkeypatch):
    # A small real module isolates checkpoint selection from backbone construction.
    monkeypatch.setattr(clay, "Encoder", lambda **kwargs: torch.nn.Linear(2, 2))

    def download(**kwargs):
        pytest.fail("an explicit checkpoint must not access the Hub")

    monkeypatch.setattr(clay, "hf_hub_download", download)
    expected = torch.nn.Linear(2, 2)
    path = tmp_path / "clay.ckpt"
    torch.save(
        {
            "state_dict": {
                **{
                    f"model.encoder.{name}": value
                    for name, value in expected.state_dict().items()
                },
                "model.decoder.unused": torch.zeros(1),
            }
        },
        path,
    )

    loaded = clay.build_clay(
        "clay_v15_tiny", pretrained=True, checkpoint_path=str(path)
    )
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_unpublished_variant_without_local_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="no published checkpoint"):
        clay.build_clay("clay_v15_tiny", pretrained=True)


def test_the_head_resizes_logits_to_the_chip_the_encoder_reads():
    from geosave.model.registry import build_model

    chain = build_model(
        {
            "encoder": {
                "name": "clay",
                "init_args": {
                    "model_name": "clay_v15_tiny",
                    "in_channels": 2,
                    "input_size": 32,
                    "waves": [0.49, 0.665],
                    "gsd": 10.0,
                },
            },
            "decoder": {"name": "unet"},
            "head": {"name": "segmentation", "init_args": {"num_classes": 2}},
        }
    ).eval()

    with torch.inference_mode():
        logits = chain(image=torch.zeros(1, 2, 32, 32))

    assert logits.shape == (1, 2, 32, 32)


def test_model_context_reads_the_raster_it_encodes(scene):
    context = clay.model_context(scene.isel(time=0))
    torch.testing.assert_close(context["time"], torch.tensor([1.0, 14.0]))
    torch.testing.assert_close(context["latlon"], torch.tensor([51.0, 11.0]))


def test_clay_refuses_multiple_frames(scene: Dataset) -> None:
    with pytest.raises(ValueError, match="single frame"):
        clay.time(scene)
