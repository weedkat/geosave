from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from geosave.cli.core.copy import safe_copy

from .templates import COMMON_DIR, TASKS_DIR, get_methods, get_tasks

_REQUIRED_DIRS = frozenset(
    ("artifacts", "release", "configs", "data", "modules", "notebooks", "scripts")
)
_EXCLUDE = frozenset(("__pycache__", ".ipynb_checkpoints", "description.txt"))


def has_method(root: Path, task: str, method: str) -> bool:
    """Return whether a workspace already contains this recipe."""
    package = root / "modules" / method.replace("-", "_")
    return package.exists() or (root / f"configs/{task}-{method}.yaml").exists()


def project_task(root: Path) -> str | None:
    """Read the workspace's single AI task, if one has been selected."""
    path = root / "project.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())["task"]


def create_workspace(root: Path) -> None:
    """Create a workspace holding shared startup files, preserving existing files.

    Args:
        root: Destination workspace directory.
    """
    for name in _REQUIRED_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    safe_copy(COMMON_DIR, root, exclude=_EXCLUDE, skip_all=True)


def add_task(root: Path, task: str, method: str) -> None:
    """Add a method for the workspace's single AI task.

    Args:
        root: Workspace directory.
        task: Bundled task name.
        method: Method belonging to the task.

    Raises:
        ValueError: The recipe is invalid, already exists, or belongs to another task.
    """
    if task not in get_tasks() or method not in get_methods(task):
        raise ValueError(f"Invalid training recipe: {task}/{method}.")
    selected = project_task(root)
    if selected is not None and selected != task:
        raise ValueError(
            f"This project owns {selected!r}; create a separate project for {task!r}."
        )
    if has_method(root, task, method):
        raise ValueError(f"{task}/{method} already exists in {root}.")
    # The catalogue groups recipes by task; a generated project already owns
    # that task, so its packages and scripts need only the method name.
    with TemporaryDirectory(prefix="geosave-recipe-") as directory:
        recipe = Path(directory)
        safe_copy(TASKS_DIR / task / method, recipe, exclude=_EXCLUDE)
        for folder in ("modules", "scripts"):
            task_folder = recipe / folder / task
            if not task_folder.exists():
                continue
            for child in task_folder.iterdir():
                child.rename(recipe / folder / child.name)
            task_folder.rmdir()

        for path in recipe.rglob("*"):
            if not path.is_file():
                continue
            content = path.read_text()
            for folder in ("modules", "scripts"):
                content = content.replace(f"{folder}/{task}/", f"{folder}/")
                content = content.replace(f"{folder}.{task}.", f"{folder}.")
            if path.suffix == ".py" and "scripts" in path.relative_to(recipe).parts:
                content = content.replace(".parents[3]", ".parents[2]")
            path.write_text(content)
        safe_copy(recipe, root, exclude=_EXCLUDE)

    if selected is None:
        (root / "project.yaml").write_text(f"task: {task}\ndata: data\n")
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {root.name}\n\n"
            f"AI task: **{task}**. Methods live in `modules/<method>/`; "
            "experiment configurations live in `configs/`.\n\n"
            "This project owns one data preparation workflow under `data/`. "
            "Document its sources, inputs, targets and splits in `data/README.md`.\n\n"
            "Describe the prediction inputs, outputs, evaluation and experiment "
            "results here. `project.yaml` records the task and data directory.\n\n"
            "Training outputs live in `artifacts/<name>/version_N/`, including "
            "TensorBoard events, the resolved config and `checkpoints/`. "
            "Leave checkpoint `dirpath` unset to follow the logger. "
            "Optional local model exports belong in `release/`.\n"
        )
