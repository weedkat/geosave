"""Portable declarations owned by ``model_spec.yaml``."""

from .base import Name, SpecModel, Text
from .call import CallSpec, CallValue, Ref
from .model import ModelSpec
from .stage import StageSpec
from .tensors import TensorRequirement
from .rasters import (
    AttrsRequirement,
    FieldRequirement,
    RasterRequirement,
)
from .stac import QueryConfig, SortConfig, StacRecipe

__all__ = [
    "AttrsRequirement",
    "CallSpec",
    "CallValue",
    "FieldRequirement",
    "ModelSpec",
    "Name",
    "QueryConfig",
    "RasterRequirement",
    "Ref",
    "SpecModel",
    "StageSpec",
    "StacRecipe",
    "SortConfig",
    "Text",
    "TensorRequirement",
]
