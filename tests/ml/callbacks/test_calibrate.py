import torch

from geosave.ml.callbacks.calibrate import softmax_argmax


def test_top_class_and_its_confidence_leave_logits_unchanged():
    logits = torch.tensor([[[[3.0, 0.0, 0.0]], [[0.0, 0.0, 3.0]]]])
    original = logits.clone()

    labels, confidence = softmax_argmax(logits)

    assert labels.tolist() == [[[0, 0, 1]]]
    torch.testing.assert_close(confidence, logits.softmax(dim=1).amax(dim=1))
    torch.testing.assert_close(logits, original)
