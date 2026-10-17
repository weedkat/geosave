import pytest
import torch

from geosave.model.spec.model import TransformSpec


def test_named_transforms_normalize_each_instant_and_preserve_context():
    from geosave.ml.transforms import InputTransform

    transform = InputTransform(
        {
            "image": [
                TransformSpec(
                    name="Normalize", init_args={"mean": [1.0, 2.0], "std": [2.0, 4.0]}
                )
            ]
        },
        size=8,
    )
    image = torch.full((2, 3, 2, 8, 8), 5.0)
    context = torch.ones(2, 2)

    result = transform({"image": image, "location_coords": context})

    assert result["location_coords"] is context
    assert result["image"].shape == image.shape
    assert (result["image"][:, :, 0] == 2.0).all()
    assert (result["image"][:, :, 1] == 0.75).all()
    assert (image == 5.0).all()


def test_input_transforms_do_not_repair_nonfinite_values():
    from geosave.ml.transforms import InputTransform

    image = torch.tensor([[[[torch.nan, torch.inf, -torch.inf, 0.0]]]])

    result = InputTransform({}, size=4)({"image": image})

    assert result["image"] is image
    assert result["image"].isnan()[0, 0, 0, 0]
    assert result["image"][0, 0, 0, 1] == torch.inf
    assert result["image"][0, 0, 0, 2] == -torch.inf


def test_per_input_transforms_reject_geometry_changes():
    from geosave.ml.transforms import InputTransform

    with pytest.raises(ValueError, match="geometry"):
        InputTransform({"image": [TransformSpec(name="RandomHorizontalFlip")]}, size=8)
