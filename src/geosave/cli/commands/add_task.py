from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import questionary as qu
import typer

from ..core.prompts import prompt_select
from ..core.templates import TASKS_DIR, get_tasks
from ..core.workspace import add_task as copy_task


def _description(task: str) -> str:
    """Read a task template's `description.txt`, or a placeholder."""
    file = TASKS_DIR / task / "description.txt"
    return file.read_text().strip() if file.exists() else "No description available."


def add_task(
    task: Annotated[
        Optional[str],
        typer.Argument(help="Task template, such as segmentation-supervised."),
    ] = None,
) -> None:
    """Add a task's configs and modules to the current workspace."""
    root = Path.cwd()
    if not (root / "main.py").exists():
        raise typer.BadParameter(
            f"main.py not found in {root}; run inside a workspace."
        )

    tasks = get_tasks()
    if task is None:
        task = prompt_select(
            "Select a task:",
            [qu.Choice(t, value=t, description=_description(t)) for t in tasks],
        )
    elif task not in tasks:
        raise typer.BadParameter(f"Task {task!r} is not a valid task.")
    copy_task(root, task)
