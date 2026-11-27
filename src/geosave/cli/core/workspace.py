"""Copy a bundled starter into a user-owned directory."""

import shutil
from contextlib import ExitStack, suppress
from pathlib import Path

from .templates import starter_entries


def create_workspace(
    root: Path, task: str = "blank", method: str | None = None, *, force: bool = False
) -> None:
    """Copy shared startup, task processing, and the selected training kit.

    Args:
        root: New or empty destination directory.
        task: Catalog task name, or `blank` for shared startup alone.
        method: Selected method for a task.
        force: Permit unrelated files already in the destination.

    Raises:
        ValueError: Unknown starter or overlapping files in the catalog.
        FileExistsError: Destination is occupied or a starter path would collide.
    """
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FileExistsError(f"{root} is not a destination directory.")
    if root.is_dir() and not force and any(root.iterdir()):
        raise FileExistsError(
            f"{root} is not empty; pass --force to allow unrelated files."
        )

    entries = starter_entries(task, method)
    conflicts = []
    for relative, source in sorted(entries.items()):
        destination = root / relative
        if destination.is_symlink() or (
            destination.exists() and not (destination.is_dir() and source.is_dir())
        ):
            conflicts.append(str(relative))
    if conflicts:
        raise FileExistsError(
            f"Existing starter paths in {root}: {', '.join(conflicts)}"
        )

    # Only roll back entries created by this invocation, including incomplete files.
    with ExitStack() as rollback:
        if not root.exists():
            root.mkdir(parents=True)
            rollback.callback(_remove_empty_directory, root)
        for relative, source in sorted(entries.items()):
            destination = root / relative
            if source.is_dir():
                if not destination.exists():
                    destination.mkdir()
                    rollback.callback(_remove_empty_directory, destination)
                continue
            with source.open("rb") as source_file, destination.open("xb") as output:
                rollback.callback(destination.unlink, missing_ok=True)
                shutil.copyfileobj(source_file, output)
            shutil.copymode(source, destination)
        rollback.pop_all()


def _remove_empty_directory(directory: Path) -> None:
    # Another writer may have placed a file here since it was created.
    with suppress(OSError):
        directory.rmdir()
