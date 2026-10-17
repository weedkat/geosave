from __future__ import annotations

from pathlib import Path

from geosave.cli.core.copy import safe_copy

from .templates import COMMON_DIR, TASKS_DIR, get_tasks

_REQUIRED_DIRS = frozenset(
    ("artifacts", "configs", "data", "modules", "notebooks", "scripts")
)
_EXCLUDE = frozenset(("__pycache__", ".ipynb_checkpoints", "description.txt"))


def create_workspace(root: Path) -> None:
    """Create a workspace holding the shared startup files.

    Args:
        root: Destination workspace directory.
    """
    for directory_name in _REQUIRED_DIRS:
        (root / directory_name).mkdir(parents=True, exist_ok=True)
    safe_copy(COMMON_DIR, root, exclude=_EXCLUDE)


def add_task(root: Path, task: str) -> None:
    """Copy one task template into an existing workspace.

    Existing files prompt before they are overwritten.

    Args:
        root: Workspace directory.
        task: Bundled task template name.

    Raises:
        ValueError: The task template does not exist.
    """
    if task not in get_tasks():
        raise ValueError(f"Task {task!r} is not a valid task.")
    safe_copy(TASKS_DIR / task, root, exclude=_EXCLUDE)
