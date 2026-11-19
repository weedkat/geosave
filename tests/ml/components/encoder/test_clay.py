import pytest
import torch
import xarray as xr
from torch.utils.data import default_collate

from geosave.geodata import Dataset
from geosave.ml.components.encoder import clay


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


def test_the_head_resizes_logits_to_the_chip_the_encoder_reads():
    from geosave.ml.components.build import build_model

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


def test_clay_reads_one_frames_week_and_hour(scene: Dataset) -> None:
    encoder = clay.Clay(
        model_name="clay_v15_tiny",
        in_channels=1,
        input_size=8,
        waves=[0.665],
        gsd=10.0,
    ).eval()
    names = ["centroid", "week", "hour"]
    frame = xr.DataTree.from_dict({"image": scene.isel(time=[0])})
    pixels, context = default_collate([frame.gs.to_tensor(context=names)])
    batch = {**pixels, **context}

    with torch.inference_mode():
        pyramid, _ = encoder.forward_pyramid(
            batch["image"][:, 0],
            centroid=batch["centroid"],
            week=batch["week"],
            hour=batch["hour"],
        )

    assert pyramid[0].shape[0] == 1
    frames = xr.DataTree.from_dict({"image": scene})
    pixels, context = default_collate([frames.gs.to_tensor(context=names)])
    batch = {**pixels, **context}
    with pytest.raises(ValueError, match="single frame"):
        encoder.forward_pyramid(
            batch["image"][:, 0], week=batch["week"], hour=batch["hour"]
        )
