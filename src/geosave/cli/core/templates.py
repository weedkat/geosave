"""Discover bundled task and method templates."""

from pathlib import Path

_EXCLUDED_TEMPLATE_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})
TEMPLATES_DIR = Path(__file__).parents[3] / "templates"
COMMON_DIR = TEMPLATES_DIR / "common"
TASKS_DIR = TEMPLATES_DIR / "tasks"


def get_tasks() -> list[str]:
    """Return task names with at least one bundled method."""
    return sorted(
        path.name
        for path in TASKS_DIR.iterdir()
        if path.is_dir()
        and path.name not in _EXCLUDED_TEMPLATE_NAMES
        and get_methods(path.name)
    )


def get_methods(task: str) -> list[str]:
    """Return the bundled methods for a task."""
    folder = TASKS_DIR / task
    if not folder.is_dir():
        return []
    methods = sorted(
        path.name
        for path in folder.iterdir()
        if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES
    )
    if "supervised" in methods:
        methods.remove("supervised")
        methods.insert(0, "supervised")
    return methods


def describe(task: str, method: str) -> str:
    """Return the method description shown during selection."""
    return (TASKS_DIR / task / method / "description.txt").read_text().strip()
