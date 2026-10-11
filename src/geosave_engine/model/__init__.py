"""What a release contains: the chain, its stages, the registry and the spec.

The chain and its builder are reached from here; each loads on first use, so
importing `geosave_engine.model.spec` never pays for torch.

Examples:
    >>> from geosave_engine.model import ModelChain, Published, build_model, chain_step
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chain import ModelChain, Published, chain_step
    from .registry import build_model

_LAZY = {
    "ModelChain": "chain",
    "Published": "chain",
    "chain_step": "chain",
    "build_model": "registry",
}

__all__ = ["ModelChain", "Published", "build_model", "chain_step"]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
