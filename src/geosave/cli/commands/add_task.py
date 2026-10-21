from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ..core.selection import select_method
from ..core.workspace import add_task as copy_task


def add_task(
    task: Annotated[
        str | None, typer.Option("--task", "-t", help="Task to add to the workspace.")
    ] = None,
    method: Annotated[
        str | None, typer.Option("--method", "-m", help="Training method for the task.")
    ] = None,
) -> None:
    """Add an editable training recipe without replacing existing methods."""
    root = Path.cwd()
    if not (root / "main.py").is_file():
        raise typer.BadParameter(
            f"main.py not found in {root}; run inside a workspace."
        )
    task, method = select_method(root, task, method, allow_blank=False)
    assert method is not None
    copy_task(root, task, method)
