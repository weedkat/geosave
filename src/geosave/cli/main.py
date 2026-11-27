import typer
from dotenv import load_dotenv

from .commands.create import create

app = typer.Typer(
    help="GeoSave CLI",
    no_args_is_help=True,
    add_completion=True,
)
app.command()(create)


@app.callback()
def global_callback() -> None:
    """Load the workspace environment before running a command."""
    load_dotenv()


if __name__ == "__main__":
    app()
