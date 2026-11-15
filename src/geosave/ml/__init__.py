"""Model chains, components, processors, releases and training support.

Public names load on first use, so `import geosave.ml` pays for neither
torch nor Lightning.

Examples:
    >>> from geosave.ml import ModelChain, build_model, chain_step, load_processor
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chain import ModelChain, chain_step
    from .components.build import build_model
    from .components.catalog import list_models
    from .components.published import Published
    from .factory import BuildSpec, Registry
    from .processing.processor import Processor, ProcessorSpec, build_processor
    from .release import load_processor

_LAZY = {
    "BuildSpec": "factory",
    "ModelChain": "chain",
    "Processor": "processing.processor",
    "ProcessorSpec": "processing.processor",
    "Published": "components.published",
    "Registry": "factory",
    "build_model": "components.build",
    "build_processor": "processing.processor",
    "chain_step": "chain",
    "list_models": "components.catalog",
    "load_processor": "release",
}

__all__ = [
    "BuildSpec",
    "ModelChain",
    "Processor",
    "ProcessorSpec",
    "Published",
    "Registry",
    "build_model",
    "build_processor",
    "chain_step",
    "list_models",
    "load_processor",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
