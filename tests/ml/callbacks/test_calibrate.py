import pytest
import torch

from geosave.ml.callbacks.calibrate import softmax_argmax


def test_prediction_decoding_is_unavailable():
    logits = torch.tensor([[[[3.0]], [[0.0]]]])
    original = logits.clone()

    with pytest.raises(NotImplementedError, match="Prediction decoding"):
        softmax_argmax(logits)

    torch.testing.assert_close(logits, original)
