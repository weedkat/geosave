"""The predictor contract.

A predictor runs a model, so it is framework code: implementations live in
that framework's package, such as `geosave.ml.torch.prediction`.
"""

from .base import ChipOutputs, PredictOutput, Predictor

__all__ = ["ChipOutputs", "PredictOutput", "Predictor"]
