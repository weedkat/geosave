"""Run GeoSave flows locally from the command line."""

from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import TypeAdapter, ValidationError
import typer

import geosave_flows
from geosave_flows.configs import AnchorConfig

_anchor_adapter = TypeAdapter(AnchorConfig)

app = typer.Typer(help="Run GeoSave flows locally.", no_args_is_help=True)


@app.callback()
def load_environment() -> None:
    """Load a .env from the current directory (or a parent), if any."""
    load_dotenv()


@app.command()
def ingest(
    anchor: Annotated[
        str,
        typer.Option(help="Anchor configuration as JSON.", metavar="JSON"),
    ],
    output: Annotated[
        Path, typer.Option(help="New folder for the rasters and their catalog.")
    ],
    spec: Annotated[Path, typer.Option(help="Model specification path.")],
) -> None:
    """Load model rasters on one anchor and write them as one catalog."""
    try:
        config = _anchor_adapter.validate_json(anchor)
    except ValidationError as error:
        raise typer.BadParameter(str(error), param_hint="--anchor") from error
    typer.echo(geosave_flows.ingest(anchor=config, output=str(output), spec=str(spec)))


if __name__ == "__main__":
    app()
