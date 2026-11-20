from __future__ import annotations

from torch import nn

from geosave.ml.criterion import ProbOhemCrossEntropy2d
from geosave.ml.factory import Registry

CRITERIA = Registry(nn.Module)
CRITERIA.register("cross_entropy")(nn.CrossEntropyLoss)
CRITERIA.register("ohem")(ProbOhemCrossEntropy2d)
