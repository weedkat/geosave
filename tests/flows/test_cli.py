import pytest
from typer.testing import CliRunner

import geosave_flows
from geosave_flows.cli import app
from geosave_flows.configs import RasterAnchorConfig

runner = CliRunner()


def test_ingest_parses_anchor_and_prints_result(
    monkeypatch,
) -> None:
    captured = {}

    def run_flow(**arguments):
        captured.update(arguments)
        return "data/raw/s1/catalog.parquet"

    monkeypatch.setattr(geosave_flows, "ingest", run_flow)
    result = runner.invoke(
        app,
        [
            "ingest",
            "--anchor",
            '{"kind":"raster","path":"data/reference.tif"}',
            "--output",
            "data/raw/s1",
            "--spec",
            "model_spec.yaml",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "anchor": RasterAnchorConfig(kind="raster", path="data/reference.tif"),
        "output": "data/raw/s1",
        "spec": "model_spec.yaml",
    }
    assert result.stdout == "data/raw/s1/catalog.parquet\n"


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

    monkeypatch.setattr(geosave_flows, "ingest", unexpected_flow)
    arguments = [
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

    monkeypatch.setattr(geosave_flows, "ingest", fail)
    result = runner.invoke(
        app,
        [
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
