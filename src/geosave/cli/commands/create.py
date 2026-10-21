from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..core.prompts import prompt_required_text
from ..core.selection import select_method
from ..core.workspace import add_task, create_workspace


def create(
    name: Annotated[
        str | None, typer.Argument(help="Directory to create the workspace in.")
    ] = None,
    task: Annotated[
        str | None, typer.Option("--task", "-t", help="Task to generate, or blank.")
    ] = None,
    method: Annotated[
        str | None, typer.Option("--method", "-m", help="Training method for the task.")
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Create inside a directory that is not empty."),
    ] = False,
) -> None:
    """Create an editable workspace; choose a task, then a training method."""
    if name is None:
        name = prompt_required_text("Enter a name for the workspace:")
    root = Path.cwd() / name
    if not force and root.is_dir() and any(root.iterdir()):
        raise typer.BadParameter(f"{root} is not empty; pass --force to use it.")
    task, method = select_method(root, task, method, allow_blank=True)
    create_workspace(root)
    if method is not None:
        add_task(root, task, method)
