"""Mean Teacher regression: an EMA teacher's weak-view values supervise strong views."""

from .data import DataModule, Unlabeled
from .module import Module

__all__ = ["DataModule", "Module", "Unlabeled"]
