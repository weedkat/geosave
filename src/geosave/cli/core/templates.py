"""Discover bundled task templates."""

from pathlib import Path

_EXCLUDED_TEMPLATE_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints"})

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
COMMON_DIR = TEMPLATES_DIR / "common"
TASKS_DIR = TEMPLATES_DIR / "tasks"


def get_tasks() -> list[str]:
    """Return the available task template names."""
    return sorted(
        path.name
        for path in TASKS_DIR.iterdir()
        if path.is_dir() and path.name not in _EXCLUDED_TEMPLATE_NAMES
    )
