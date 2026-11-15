from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from geosave.ml.criterion import ProbOhemCrossEntropy2d
from geosave.ml.factory import BuildSpec, Registry


class CriterionSpec(BuildSpec):
    """Configure a criterion module."""


CRITERIA = Registry(nn.Module)
CRITERIA.register("cross_entropy")(nn.CrossEntropyLoss)
CRITERIA.register("ohem")(ProbOhemCrossEntropy2d)


def build_criterion(spec: CriterionSpec | Mapping[str, Any]) -> nn.Module:
    """Construct a criterion from a registered name or imported class.

    Args:
        spec: Criterion selector and constructor arguments.

    Returns:
        Configured criterion module.
    """
    return CRITERIA.build(CriterionSpec.model_validate(spec))
