"""Run GeoSave workflows locally."""

import typer

from .ingest import ingest

workflow = typer.Typer(
    help="Run complete GeoSave workflows locally.",
    no_args_is_help=True,
)
workflow.command()(ingest)
