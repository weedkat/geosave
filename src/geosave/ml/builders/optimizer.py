from __future__ import annotations

from torch.optim import SGD, Adagrad, Adam, AdamW, Optimizer, RMSprop

from geosave.ml.factory import Registry

OPTIMIZERS = Registry(Optimizer)
OPTIMIZERS.register("adamw")(AdamW)
OPTIMIZERS.register("adam")(Adam)
OPTIMIZERS.register("sgd")(SGD)
OPTIMIZERS.register("rmsprop")(RMSprop)
OPTIMIZERS.register("adagrad")(Adagrad)
