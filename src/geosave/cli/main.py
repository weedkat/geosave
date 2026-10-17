import typer
from dotenv import load_dotenv

from .commands.add_task import add_task
from .commands.create import create
from .commands.publish import publish

app = typer.Typer(
    help="GeoSave CLI",
    no_args_is_help=True,
    add_completion=True,
)
app.command()(create)
app.command("add-task")(add_task)
app.command()(publish)


@app.callback()
def global_callback() -> None:
    """Load a .env from the current directory (or a parent), if any, before any command runs."""
    load_dotenv()


if __name__ == "__main__":
    app()
