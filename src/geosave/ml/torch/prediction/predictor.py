"""Run a PyTorch model over the prepared windows, chip by chip."""

import numpy as np
import torch
import xarray as xr
from torch import nn

from geosave.geodata import RasterChipper
from geosave.geodata.sampling import ChipMode
from geosave.ml.pipeline.prediction import ChipOutputs, PredictOutput, Predictor
from geosave.ml.pipeline.preprocessing import PreprocessOutput


class TorchPredictor(Predictor):
    """Run a PyTorch model over the prepared windows, chip by chip.

    Each chip's raster node paths and context keys are passed to the model by
    name. Floating-point inputs take the dtype of the model's weights; each
    chip's output is split from its batch.

    Args:
        model: The model, such as a released chain loaded with
            `ModelChain.from_pretrained`.
        chip_size: Chip height and width in pixels.
        chip_overlap: Shared pixels per axis, or a fraction of `chip_size` in [0, 1).
        chip_mode: Chip edge policy, as for `RasterChipper`.
        chip_constant_value: Fill for `chip_mode="constant"`.
        batch_size: Chips per model call.
        context: Model parameter names mapped to the names of their readers
            in `CONTEXT_READERS`, such as `{"centroid": "centroid"}`.
        device: Torch device the model runs on.

    Examples:
        >>> model = ModelChain.from_pretrained("org/landcover")
        >>> predictor = TorchPredictor(model, chip_size=224, chip_overlap=0.5)
        >>> predicted = predictor(preprocessed)
        >>> location, logits = next(predicted.outputs)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        chip_size: int,
        chip_overlap: int | float = 0,
        chip_mode: ChipMode = "reflect",
        chip_constant_value: int | float = 0,
        batch_size: int = 8,
        context: dict[str, str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.chip_size = chip_size
        self.chip_overlap = chip_overlap
        self.chip_mode: ChipMode = chip_mode
        self.chip_constant_value = chip_constant_value
        self.batch_size = batch_size
        self.context = {} if context is None else context
        self.device = device

    def __call__(self, preprocessed: PreprocessOutput) -> PredictOutput[np.ndarray]:
        """Return the chips and each chip's output as the model runs.

        Args:
            preprocessed: The preprocess stage's output.

        Returns:
            The chipper, and each chip's location with its output as a
            float32 NumPy array, window after window in the order
            `RasterChipper.iter_batches` reads them.
        """
        chips = RasterChipper(
            preprocessed.windows,
            size=self.chip_size,
            overlap=self.chip_overlap,
            mode=self.chip_mode,
            constant_value=self.chip_constant_value,
        )
        return PredictOutput(chips, self.predict_chips(chips))

    @torch.inference_mode()
    def predict_chips(
        self, chips: RasterChipper[xr.DataTree]
    ) -> ChipOutputs[np.ndarray]:
        """Yield each chip's location and output, one batch of chips at a time.

        Raises:
            TypeError: The model returns no tensor, such as a detector's
                per-image dictionaries.
        """
        dtype = next(self.model.parameters()).dtype
        for batch in chips.iter_batches(self.batch_size):
            pixels, context = batch.to_tensor(context=self.context)
            inputs = {}
            for name, value in (pixels | context).items():
                if value.is_floating_point():
                    value = value.to(dtype)
                inputs[name] = value.to(self.device)

            outputs = self.model(**inputs)
            if not isinstance(outputs, torch.Tensor):
                raise TypeError(
                    f"TorchPredictor splits tensor outputs by chip; the model "
                    f"returned {type(outputs).__name__}"
                )
            values = outputs.float().cpu().numpy()
            for location, value in zip(batch.locations, values, strict=True):
                yield location, value
