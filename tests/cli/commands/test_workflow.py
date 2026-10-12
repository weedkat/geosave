import pytest
from typer.testing import CliRunner

from geosave_engine.cli.main import app
from geosave_engine.workflow import flows
from geosave_engine.workflow.configs import RasterAnchorConfig

runner = CliRunner()


def test_workflow_help_lists_only_runnable_commands() -> None:
    root = runner.invoke(app, ["--help"])
    group = runner.invoke(app, ["workflow", "--help"])

    assert root.exit_code == 0
    assert "workflow" in root.stdout
    assert group.exit_code == 0
    assert "ingest" in group.stdout
    assert "prepare-dense-data" not in group.stdout


def test_ingest_help_lists_its_flow_options() -> None:
    result = runner.invoke(app, ["workflow", "ingest", "--help"])

    assert result.exit_code == 0
    for option in ("--anchor", "--output", "--spec", "--format"):
        assert option in result.stdout
    assert "--catalog" not in result.stdout
    assert "--sources" not in result.stdout


def test_ingest_parses_anchor_and_prints_result(
    monkeypatch,
) -> None:
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "data/scenes/s1"

    monkeypatch.setattr(flows, "ingest", run_flow)
    result = runner.invoke(
        app,
        [
            "workflow",
            "ingest",
            "--anchor",
            '{"kind":"raster","path":"data/reference.tif"}',
            "--output",
            "data/scenes/s1",
            "--spec",
            "model_spec.yaml",
            "--format",
            "geotiff",
            "--write-options",
            '{"compress":"ZSTD"}',
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "anchor": RasterAnchorConfig(kind="raster", path="data/reference.tif"),
        "output": "data/scenes/s1",
        "spec": "model_spec.yaml",
        "format": "geotiff",
        "write_options": {"compress": "ZSTD"},
    }
    assert result.stdout == "data/scenes/s1\n"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--anchor", "{"),
        ("--anchor", '{"kind":"unknown"}'),
    ],
)
def test_ingest_rejects_invalid_json_before_invoking_flow(
    monkeypatch, option, value
) -> None:
    def unexpected_flow(**arguments):
        pytest.fail(f"flow invoked with invalid input: {arguments}")

    monkeypatch.setattr(flows, "ingest", unexpected_flow)
    arguments = [
        "workflow",
        "ingest",
        "--anchor",
        '{"kind":"raster","path":"reference.tif"}',
        "--output",
        "raw",
        "--spec",
        "model_spec.yaml",
    ]
    arguments.extend([option, value])

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2


def test_flow_failure_exits_without_printing_success_path(monkeypatch) -> None:
    def fail(**arguments):
        raise RuntimeError("ingestion failed")

    monkeypatch.setattr(flows, "ingest", fail)
    result = runner.invoke(
        app,
        [
            "workflow",
            "ingest",
            "--anchor",
            '{"kind":"raster","path":"reference.tif"}',
            "--output",
            "raw",
            "--spec",
            "model_spec.yaml",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "raw" not in result.stdout
