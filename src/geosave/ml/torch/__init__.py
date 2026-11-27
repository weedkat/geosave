"""PyTorch model chains, their modules and releases.

Public names load on first use, so importing this package pays for neither
Transformers nor Lightning. Lightning training support lives in
`geosave.ml.torch.training`.

Examples:
    >>> from geosave.ml.torch import ModelChain, list_models
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chain import ModelChain, Published
    from .chain.catalog import list_models
    from .chain.registry import MODULES, BuildSpec, Registry
    from .prediction import TorchPredictor

_LAZY = {
    "BuildSpec": "chain.registry",
    "MODULES": "chain.registry",
    "ModelChain": "chain",
    "Published": "chain",
    "Registry": "chain.registry",
    "list_models": "chain.catalog",
    "TorchPredictor": "prediction",
}

__all__ = [
    "MODULES",
    "BuildSpec",
    "ModelChain",
    "Published",
    "Registry",
    "TorchPredictor",
    "list_models",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
