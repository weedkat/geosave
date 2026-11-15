"""Registries of the components a model recipe names, one per stage."""

from torch import nn

from geosave.ml.factory import Registry

ENCODERS = Registry(nn.Module, imports=("geosave.ml.components.encoder",))
DECODERS = Registry(nn.Module, imports=("geosave.ml.components.decoder",))
HEADS = Registry(nn.Module, imports=("geosave.ml.components.head",))
MONOLITHS = Registry(nn.Module, imports=("geosave.ml.components.monolith",))

# A recipe's stage name selects the registry its component name is looked up in.
COMPONENTS: dict[str, Registry[nn.Module]] = {
    "encoder": ENCODERS,
    "decoder": DECODERS,
    "head": HEADS,
    "model": MONOLITHS,
}
