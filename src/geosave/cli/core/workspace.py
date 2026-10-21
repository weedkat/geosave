from __future__ import annotations

from pathlib import Path

from geosave.cli.core.copy import safe_copy

from .templates import COMMON_DIR, TASKS_DIR, get_methods, get_tasks

_REQUIRED_DIRS = frozenset(
    ("artifacts", "configs", "data", "modules", "notebooks", "scripts")
)
_EXCLUDE = frozenset(("__pycache__", ".ipynb_checkpoints", "description.txt"))


def has_method(root: Path, task: str, method: str) -> bool:
    """Return whether a workspace already contains this recipe."""
    package = root / "modules" / task / method.replace("-", "_")
    return package.exists() or (root / f"configs/{task}-{method}.yaml").exists()


def create_workspace(root: Path) -> None:
    """Create a workspace holding shared startup files, preserving existing files.

    Args:
        root: Destination workspace directory.
    """
    for name in _REQUIRED_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    safe_copy(COMMON_DIR, root, exclude=_EXCLUDE, skip_all=True)


def add_task(root: Path, task: str, method: str) -> None:
    """Copy an independent training recipe into a workspace.

    Args:
        root: Workspace directory.
        task: Bundled task name.
        method: Method belonging to the task.

    Raises:
        ValueError: The recipe is invalid or already exists.
    """
    if task not in get_tasks() or method not in get_methods(task):
        raise ValueError(f"Invalid training recipe: {task}/{method}.")
    if has_method(root, task, method):
        raise ValueError(f"{task}/{method} already exists in {root}.")
    safe_copy(TASKS_DIR / task / method, root, exclude=_EXCLUDE)
