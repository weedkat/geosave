"""Choose a starter from the installed catalog."""

import questionary as qu
import typer

from .prompts import prompt_select
from .templates import get_methods, get_tasks, read_method_description


def select_method(task: str | None, method: str | None) -> tuple[str, str | None]:
    """Prompt for missing choices, automatically selecting a task's sole method."""
    tasks = ["blank", *get_tasks()]
    if task is None:
        task = prompt_select("Select a task:", tasks)
    if task not in tasks:
        raise typer.BadParameter(
            f"Unknown task {task!r}. Choose from {', '.join(tasks)}."
        )
    if task == "blank":
        if method is not None:
            raise typer.BadParameter("A blank workspace has no training method.")
        return task, None

    methods = get_methods(task)
    if method is None:
        if len(methods) == 1:
            return task, methods[0]
        choices = [
            qu.Choice(name, value=name, description=read_method_description(task, name))
            for name in methods
        ]
        method = prompt_select(f"Select a training method for {task}:", choices)
    if method not in methods:
        raise typer.BadParameter(
            f"Unknown method {method!r} for {task!r}. Choose from {', '.join(methods)}."
        )
    return task, method
