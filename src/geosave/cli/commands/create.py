from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from ..core.prompts import prompt_required_text
from ..core.workspace import create_workspace


def create(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Directory to create the workspace in."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Create inside a directory that is not empty."),
    ] = False,
) -> None:
    """Create an empty GeoSave workspace; `add-task` fills it."""
    if name is None:
        name = prompt_required_text("Enter a name for the workspace:")
    root = Path.cwd() / name
    if not force and root.is_dir() and any(root.iterdir()):
        raise typer.BadParameter(f"{root} is not empty; pass --force to use it.")
    create_workspace(root)
