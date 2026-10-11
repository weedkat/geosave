"""Mean Teacher regression: an EMA teacher's weak-view values supervise strong views."""

from .data import DataModule
from .module import Module

__all__ = ["DataModule", "Module"]
