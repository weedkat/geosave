"""Select a task and one of its training methods."""

from pathlib import Path

import questionary as qu
import typer

from .prompts import prompt_select
from .templates import describe, get_methods, get_tasks
from .workspace import has_method


def select_method(
    root: Path, task: str | None, method: str | None, *, allow_blank: bool
) -> tuple[str, str | None]:
    """Choose missing task/method options and validate before writing files.

    Args:
        root: Destination workspace, used to disable existing methods.
        task: Explicit task, or None to prompt.
        method: Explicit method, or None to prompt.
        allow_blank: Whether shared startup alone is an available choice.

    Returns:
        Task and method, with None as the method for a blank workspace.

    Raises:
        typer.BadParameter: A selection is invalid or already exists.
    """
    tasks = get_tasks()
    if allow_blank:
        tasks.insert(0, "blank")
    if task is None:
        task = prompt_select("Select a task:", tasks)
    if task not in tasks:
        raise typer.BadParameter(f"Task {task!r} is not a valid task.")
    if task == "blank":
        if method is not None:
            raise typer.BadParameter("A blank workspace has no training method.")
        return task, None

    methods = get_methods(task)
    if method is None:
        choices = []
        for name in methods:
            exists = has_method(root, task, name)
            choices.append(
                qu.Choice(
                    name,
                    value=name,
                    description=describe(task, name),
                    disabled="already exists" if exists else None,
                )
            )
        if all(has_method(root, task, name) for name in methods):
            raise typer.BadParameter(f"All methods for {task!r} already exist.")
        method = prompt_select(f"Select a training method for {task}:", choices)
    if method not in methods:
        raise typer.BadParameter(f"Method {method!r} is not valid for task {task!r}.")
    if has_method(root, task, method):
        raise typer.BadParameter(f"{task}/{method} already exists in {root}.")
    return task, method
