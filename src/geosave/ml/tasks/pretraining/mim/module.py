from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from geosave.ml.builders import build_optimizer, build_scheduler
from geosave.model.chain import ModelChain
from geosave.model.registry import build_model

from .data import Sample


def patch_mask(
    batch: int,
    shape: tuple[int, int],
    *,
    patch_size: int,
    ratio: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mask a random share of square patches in every sample.

    Args:
        batch: Number of samples.
        shape: Height and width of the samples.
        patch_size: Side of a masked patch in pixels.
        ratio: Share of patches masked in each sample.
        device: Device of the returned mask.
        generator: Random source; the generator's device must be `device`.

    Returns:
        Boolean ``(batch, height, width)`` mask, true where pixels are hidden.

    Examples:
        >>> patch_mask(2, (64, 64), patch_size=16, ratio=0.5).float().mean()
        tensor(0.5000)
    """
    height, width = shape
    rows = -(-height // patch_size)
    columns = -(-width // patch_size)
    count = rows * columns
    # The lowest random scores pick the masked patches.
    chosen = torch.rand(batch, count, device=device, generator=generator).argsort(
        dim=1
    )[:, : round(ratio * count)]
    grid = torch.zeros(batch, count, dtype=torch.bool, device=device)
    grid.scatter_(1, chosen, True)
    grid = grid.reshape(batch, rows, columns)
    pixels = grid.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
    return pixels[:, :height, :width]


class Module(LightningModule):
    """Pretrain a model chain by reconstructing masked patches of its input.

    Hidden pixels are scored by mean absolute error. The last stage
    predicts one value per band and instant, as a regression head naming them.

    Args:
        model_chain: Ordered stage construction specifications.
        optimizer: Optimizer class, arguments, and groups by stage.
            Defaults to AdamW with a learning rate of 1e-4.
        lr_scheduler: Scheduler class, arguments, and Lightning metadata.
        reconstruct: Pixel input that is masked and reconstructed.
        patch_size: Side of a masked patch in pixels; match the encoder's
            patch size for a ViT.
        mask_ratio: Share of patches hidden in each sample.

    Examples:
        # LightningCLI YAML:
        model:
          class_path: geosave.ml.tasks.pretraining.mim.Module
          init_args:
            mask_ratio: 0.6
            patch_size: 16
            model_chain:
              encoder: {name: dinov3, init_args: {in_channels: 4}}
              decoder: {name: unet}
              head: {name: regression, init_args: {variables: [blue, green, red, nir]}}
    """

    model: ModelChain

    def __init__(
        self,
        *,
        model_chain: dict[str, dict[str, Any]],
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        reconstruct: str = "image",
        patch_size: int = 16,
        mask_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        if not model_chain:
            raise ValueError("Supply at least one model stage")
        if optimizer is None:
            optimizer = {"name": "adamw", "init_args": {"lr": 1e-4}}
        self.save_hyperparameters()
        self.model_chain = model_chain
        self.optimizer_spec = deepcopy(optimizer)
        self.scheduler_spec = deepcopy(lr_scheduler)
        self.reconstruct = reconstruct
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def configure_model(self) -> None:
        """Construct the model chain."""
        if hasattr(self, "model"):
            return
        self.model = build_model(deepcopy(self.model_chain))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build optimization for the configured chain's trainable parameters.

        Returns:
            The optimizer alone when scheduling is disabled, otherwise a
            Lightning dictionary with `optimizer` and `lr_scheduler` entries.
        """
        optimizer = build_optimizer(self.optimizer_spec, self.model)
        if self.scheduler_spec is None:
            return optimizer
        return cast(
            OptimizerLRScheduler,
            {
                "optimizer": optimizer,
                "lr_scheduler": build_scheduler(self.scheduler_spec, optimizer),
            },
        )

    def forward(self, **model_inputs: Any) -> torch.Tensor:
        """Return the chain's per-pixel reconstruction of prepared inputs.

        Args:
            **model_inputs: Batched, prepared pixels and context under the
                configured chain's argument names.

        Returns:
            `(batch, channels, y, x)` reconstruction. `channels` covers every
            band and instant of the reconstructed pixel input, flattened in
            their input order.

        Raises:
            TypeError: The model does not return one tensor.
        """
        reconstruction = self.model(**model_inputs)
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("A reconstruction model must return one tensor")
        return reconstruction

    def _loss(self, batch: Sample, seed: int | None = None) -> torch.Tensor:
        """Mask the input, reconstruct it, and score the hidden pixels.

        Args:
            batch: A finished `Sample` batch.
            seed: Seed of the mask. None draws a fresh one.

        Raises:
            ValueError: The reconstruction does not hold one channel per band
                and instant of the input.
        """
        inputs = batch["inputs"]
        image = inputs[self.reconstruct]
        height, width = image.shape[-2:]
        generator = None
        if seed is not None:
            generator = torch.Generator(image.device).manual_seed(seed)
        mask = patch_mask(
            len(image),
            (height, width),
            patch_size=self.patch_size,
            ratio=self.mask_ratio,
            device=image.device,
            generator=generator,
        )
        # One mask hides a patch in every instant and band of a sample.
        spread = mask.reshape(len(mask), *([1] * (image.ndim - 3)), height, width)
        hidden = image.masked_fill(spread, 0.0)
        reconstruction = self(**{**inputs, self.reconstruct: hidden})

        target = image.flatten(1, -3)
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"the reconstruction is {tuple(reconstruction.shape)}, the "
                f"input {self.reconstruct!r} flattened to bands is "
                f"{tuple(target.shape)}; the last stage predicts one value per "
                f"band and instant"
            )
        scored = mask.float()
        error = (reconstruction - target).abs().mean(dim=1)
        return (error * scored).sum() / scored.sum().clamp(min=1)

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        """Draw a fresh patch mask and log reconstruction loss on hidden pixels.

        Args:
            batch: Prepared `Sample` holding named pixel/context inputs and
                without class targets.
            batch_idx: Lightning's batch position; unused for the random mask.

        Returns:
            Scalar mean absolute error over hidden pixels, averaged
            across reconstructed channels. No scored pixels yields zero.
        """
        loss = self._loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["inputs"][self.reconstruct]),
        )
        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        """Log reconstruction loss using a reproducible mask for this batch.

        Args:
            batch: Prepared `Sample` with named inputs.
            batch_idx: Seed for the validation mask, keeping the mask fixed
                for the same batch position across epochs.
        """
        # A mask seeded by batch keeps val_loss comparable across epochs.
        self.log(
            "val_loss",
            self._loss(batch, seed=batch_idx),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["inputs"][self.reconstruct]),
        )
