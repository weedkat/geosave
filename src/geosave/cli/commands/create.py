from pathlib import Path
from typing import Annotated

import typer

from ..core.prompts import prompt_required_text
from ..core.selection import select_method
from ..core.workspace import create_workspace


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
        typer.Option("--force", help="Allow unrelated files in the destination."),
    ] = False,
) -> None:
    """Create an editable workspace from one task and training method."""
    if name is None:
        name = prompt_required_text("Enter a name for the workspace:")
    task, method = select_method(task, method)
    try:
        create_workspace(Path(name), task, method, force=force)
    except FileExistsError as error:
        raise typer.BadParameter(str(error)) from error
