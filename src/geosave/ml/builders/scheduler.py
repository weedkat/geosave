from __future__ import annotations

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

from geosave.ml.factory import Registry

SCHEDULERS = Registry(LRScheduler)
SCHEDULERS.register("cosine_annealing")(CosineAnnealingLR)
SCHEDULERS.register("reduce_on_plateau")(ReduceLROnPlateau)
SCHEDULERS.register("step")(StepLR)
