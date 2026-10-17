from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from geosave.ml.tasks.pretraining import mim
from geosave.ml.tasks.pretraining.mim.module import patch_mask
from geosave.model.chain import chain_step


class Reconstructor(nn.Module):
    """Return the visible pixels, bands and instants flattened, times a scale."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def reconstruct(self, image: torch.Tensor) -> torch.Tensor:
        return image.flatten(1, -3) * self.scale


STAGES = {"model": {"class_path": f"{__name__}.Reconstructor"}}


def test_patch_masks_hide_whole_patches_in_the_given_share() -> None:
    mask = patch_mask(3, (32, 48), patch_size=8, ratio=0.25)

    assert mask.shape == (3, 32, 48)
    patches = mask.reshape(3, 4, 8, 6, 8).permute(0, 1, 3, 2, 4).flatten(3)
    # Each patch is hidden whole or not at all.
    assert torch.equal(patches.all(dim=-1), patches.any(dim=-1))
    assert patches.all(dim=-1).flatten(1).sum(dim=1).tolist() == [6, 6, 6]


def test_patch_masks_crop_patches_reaching_past_the_edge() -> None:
    mask = patch_mask(1, (10, 10), patch_size=4, ratio=1.0)

    assert mask.shape == (1, 10, 10) and mask.all()


def _module(**options) -> mim.Module:
    module = mim.Module(model_chain=STAGES, **options)
    module.configure_model()
    return module


@pytest.mark.parametrize(("ratio", "expected"), [(0.5, 1.0), (0.0, 0.0)])
def test_only_hidden_pixels_are_scored(ratio: float, expected: float) -> None:
    module = _module(patch_size=2, mask_ratio=ratio)
    image = torch.ones(2, 3, 2, 8, 8)

    loss = module.training_step({"inputs": {"image": image}}, 0)

    assert loss.item() == pytest.approx(expected)


def test_a_reconstruction_must_cover_every_band_and_instant() -> None:
    module = _module()
    module.model.get_submodule("model").reconstruct = lambda image: image[:, 0]  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="one value per band"):
        module.training_step(
            {
                "inputs": {"image": torch.ones(1, 2, 16, 16)},
            },
            0,
        )


class _Chips(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {
            "inputs": {"image": torch.randn(2, 8, 8)},
        }


def test_lightning_fits_and_validates(tmp_path) -> None:
    module = mim.Module(model_chain=STAGES, patch_size=4)
    trainer = Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    loader = DataLoader(_Chips(), batch_size=2)

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert {"train_loss", "val_loss"} <= set(trainer.callback_metrics)


def test_a_validation_batch_hides_the_same_patches_every_epoch() -> None:
    module = _module(patch_size=2, mask_ratio=0.5)
    image = torch.rand(2, 2, 8, 8)
    batch: mim.Sample = {
        "inputs": {"image": image},
    }

    first = [module._loss(batch, seed=index) for index in (0, 1)]
    again = [module._loss(batch, seed=index) for index in (0, 1)]

    assert torch.equal(torch.stack(first), torch.stack(again))
    assert not torch.equal(first[0], first[1])
