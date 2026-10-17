import pytest
import torch

from geosave.ml.transforms import cutmix_boxes


def test_each_box_is_one_rectangle_within_the_area_bounds() -> None:
    torch.manual_seed(0)

    boxes = cutmix_boxes(64, (32, 48), scale=(0.1, 0.3))

    assert boxes.shape == (64, 32, 48)
    assert boxes.dtype == torch.bool
    for box in boxes:
        rows = box.any(dim=1).nonzero().squeeze(1)
        columns = box.any(dim=0).nonzero().squeeze(1)
        height = int(rows[-1] - rows[0] + 1)
        width = int(columns[-1] - columns[0] + 1)
        # A filled bounding rectangle: no holes inside the box.
        assert int(box.sum()) == height * width
        assert 0.05 <= height * width / (32 * 48) <= 0.35


@pytest.mark.parametrize(("p", "expected"), [(0.0, 0), (1.0, 16)])
def test_probability_decides_which_samples_get_a_box(p: float, expected: int) -> None:
    boxes = cutmix_boxes(16, (8, 8), p=p)

    assert int(boxes.flatten(1).any(dim=1).sum()) == expected


def test_cutmix_pastes_one_donor_box_across_temporal_inputs_and_targets():
    from geosave.ml.transforms import CutMix

    image = torch.arange(2 * 3 * 2 * 8 * 8, dtype=torch.float32).reshape(2, 3, 2, 8, 8)
    target = image[:, 0, 0].long()
    dem = target[:, None].float()

    pixels, elevation, labels = CutMix(p=1.0)(image, dem, target)

    changed = (pixels != image).flatten(1, 2).all(dim=1)
    assert changed.any()
    assert torch.equal(labels, torch.where(changed, target.roll(1, 0), target))
    assert torch.equal(elevation[:, 0].long(), labels)
    torch.testing.assert_close(
        pixels, torch.where(changed[:, None, None], image.roll(1, 0), image)
    )
