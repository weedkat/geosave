"""Find per-class confidence thresholds for a release's postprocessor."""

from __future__ import annotations

from typing import Literal

import torch


def calibrate_thresholds(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
    metric: Literal["f1", "iou"],
    steps: int,
) -> list[float]:
    """Find the confidence threshold per class that maximises a metric.

    Not implemented. The intended calibration uses the best checkpoint's
    validation predictions. A future postprocessor would store the thresholds
    in its settings and leave low-confidence pixels unclassified; the current
    segmentation postprocessor uses argmax and accepts no thresholds.

    Args:
        probabilities: Class probabilities shaped `(batch, class, y, x)`.
        labels: int64 labels shaped `(batch, y, x)`.
        ignore_index: Label left out of the metric.
        metric: Metric each class's threshold maximises.
        steps: Candidate thresholds tried per class, evenly spaced in `[0, 1]`.

    Returns:
        One threshold per class, in class order.

    Raises:
        NotImplementedError: Threshold calibration is in progress.
    """
    raise NotImplementedError("Threshold calibration is in progress.")
