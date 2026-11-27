"""Discover the task starters shipped with GeoSave."""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parents[3] / "templates"
COMMON_DIR = TEMPLATES_DIR / "common"
TASKS_DIR = TEMPLATES_DIR / "tasks"


def get_tasks() -> list[str]:
    """Return the task names that offer training methods."""
    return sorted(task.name for task in TASKS_DIR.iterdir() if get_methods(task.name))


def get_methods(task: str) -> list[str]:
    """Return this task's method names, listing supervised first when available."""
    descriptions = (TASKS_DIR / task / "methods").glob("*/description.txt")
    return sorted(
        (path.parent.name for path in descriptions if path.is_file()),
        key=lambda name: (name != "supervised", name),
    )


def read_method_description(task: str, method: str) -> str:
    """Return the description shown in the method selector."""
    path = TASKS_DIR / task / "methods" / method / "description.txt"
    return path.read_text().strip()


def starter_entries(task: str, method: str | None) -> dict[Path, Path]:
    """Map destination-relative names to the bundled files and directories to copy."""
    sources = [COMMON_DIR]
    if task == "blank":
        if method is not None:
            raise ValueError("A blank workspace has no training method.")
    else:
        if task not in get_tasks() or method is None or method not in get_methods(task):
            raise ValueError(f"Unknown training starter: {task}/{method}.")
        task_common = TASKS_DIR / task / "common"
        if task_common.is_dir():
            sources.append(task_common)
        sources.append(TASKS_DIR / task / "methods" / method)

    entries: dict[Path, Path] = {}
    for source in sources:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if {"__pycache__", ".ipynb_checkpoints"}.intersection(relative.parts):
                continue
            if relative == Path("description.txt"):
                continue
            previous = entries.get(relative)
            if previous is not None and not (previous.is_dir() and path.is_dir()):
                raise ValueError(f"Starter sources overlap at {relative}.")
            entries[relative] = path
    return entries
