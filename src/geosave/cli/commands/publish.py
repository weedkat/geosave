from pathlib import Path
from typing import Annotated, Optional

import typer


def publish(
    checkpoint: Annotated[
        Optional[Path],
        typer.Argument(help="Lightning checkpoint to release; prompted when omitted."),
    ] = None,
    repo: Annotated[
        Optional[str],
        typer.Option("-r", "--repo", help="Hugging Face repo id, org/name."),
    ] = None,
) -> None:
    """Release one checkpoint as an inference model on the Hugging Face Hub."""
    typer.echo(
        "geosave publish is not implemented yet; "
        "use scripts/publish.py from the segmentation starter.",
        err=True,
    )
    raise typer.Exit(1)
