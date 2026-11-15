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

    Status: in progress. Run it after training on the best checkpoint's
    validation predictions, and give the result to the postprocessor's
    `init_args`; the trained model itself stores no thresholds. A pixel whose
    top-class probability falls below its class's threshold is left unclassified.

    Args:
        probabilities: Class probabilities shaped `(batch, class, y, x)`.
        labels: int64 labels shaped `(batch, y, x)`.
        ignore_index: Label left out of the metric.
        metric: Metric each class's threshold maximises.
        steps: Candidate thresholds tried per class, evenly spaced in `[0, 1]`.

    Returns:
        One threshold per class, in class order.

    Examples:
        >>> thresholds = calibrate_thresholds(
        ...     probabilities, labels, ignore_index=255, metric="f1", steps=100
        ... )
        >>> postprocessor = {"name": "classes", "init_args": {"thresholds": thresholds}}
    """
    raise NotImplementedError("Threshold calibration is in progress.")
