"""Contract for running a model over the prepared windows, and the predict stage's output."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import xarray as xr

from geosave.geodata.sampling import ChipLocation, RasterChipper
from geosave.ml.pipeline.preprocessing import PreprocessOutput

type ChipOutputs[Output] = Iterator[tuple[ChipLocation, Output]]


@dataclass(frozen=True)
class PredictOutput[Output]:
    """What the predict stage hands the postprocessor: each chip's model output.

    Attributes:
        chips: The chipper that cut the windows, so a postprocessor knows
            every window's grid and dates, such as to build a `RasterMerger`.
        outputs: Each chip's location with its output, produced as the model
            runs; read once.
        metadata: Values a custom stage adds for a later one.
    """

    chips: RasterChipper[xr.DataTree]
    outputs: ChipOutputs[Output]
    metadata: dict[str, Any] = field(default_factory=dict)


class Predictor(ABC):
    """A runner of one framework's model over the prepared windows.

    The prepared trees' raster node paths name the model's inputs. A
    predictor cuts every window into model-sized chips and yields each
    chip's output with its location; a postprocessor turns the outputs into
    the product.

    A predictor takes its model as the constructor argument `model` and
    keeps it under that name. A pipeline release saves that model with the
    model's own `save_pretrained` and loads it back with its class's
    `from_pretrained`.

    Attributes:
        model: The model this predictor runs.
    """

    model: Any

    @abstractmethod
    def __call__(self, preprocessed: PreprocessOutput) -> PredictOutput[Any]:
        """Return the chips and each chip's model output, window after window.

        Args:
            preprocessed: The preprocess stage's output.

        Returns:
            The chipper and each chip's output in the structure the model
            returns, tensors as NumPy arrays.
        """
