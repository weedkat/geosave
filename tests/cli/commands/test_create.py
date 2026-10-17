"""Workspace creation and task templates through the public CLI."""

import pytest
from typer.testing import CliRunner

from geosave.cli.core import copy
from geosave.cli.main import app

TASKS = ["segmentation-supervised", "regression-supervised", "lightning-boilerplate"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["create", "demo"])
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(tmp_path / "demo")
    return tmp_path / "demo"


def test_create_makes_an_empty_workspace(workspace):
    assert (workspace / "main.py").is_file()
    assert (workspace / ".gitignore").is_file()
    assert (workspace / ".env.example").is_file()
    assert not (workspace / ".env").exists()
    assert not (workspace / "geosave.toml").exists()
    assert not (workspace / "logs").exists()
    assert not (workspace / "predictions").exists()
    assert (workspace / "artifacts").is_dir()
    assert not any((workspace / "configs").iterdir())


def test_create_refuses_non_empty_directory_unless_forced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo/notes.txt").write_text("keep")
    refused = CliRunner().invoke(app, ["create", "demo"])
    assert refused.exit_code == 2
    assert not (tmp_path / "demo/main.py").exists()
    forced = CliRunner().invoke(app, ["create", "demo", "--force"])
    assert forced.exit_code == 0, forced.output
    assert (tmp_path / "demo/main.py").is_file()
    assert (tmp_path / "demo/notes.txt").read_text() == "keep"


@pytest.mark.parametrize("task", TASKS)
def test_add_task_writes_the_setup_config(workspace, task):
    result = CliRunner().invoke(app, ["add-task", task])
    assert result.exit_code == 0, result.output
    assert (workspace / f"configs/{task}.yaml").is_file()
    assert not (workspace / "description.txt").exists()
    assert (workspace / "modules/datasets.py").is_file() == task.endswith("supervised")


def test_second_task_keeps_the_shared_files(workspace, monkeypatch):
    runner = CliRunner()
    runner.invoke(app, ["add-task", "segmentation-supervised"])
    (workspace / "modules/datasets.py").write_text("mine = True\n")
    monkeypatch.setattr(copy, "_ask_overwrite", lambda path: "skip")
    runner.invoke(app, ["add-task", "regression-supervised"])
    assert (workspace / "configs/regression-supervised.yaml").is_file()
    assert (workspace / "modules/datasets.py").read_text() == "mine = True\n"


def test_add_task_rejects_unknown_task(workspace):
    result = CliRunner().invoke(app, ["add-task", "missing"])
    assert result.exit_code == 2
    assert "not a valid task" in result.output


def test_add_task_requires_a_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["add-task", "segmentation-supervised"])
    assert result.exit_code == 2
    assert not (tmp_path / "configs").exists()


def test_publish_reports_not_implemented():
    result = CliRunner().invoke(app, ["publish", "run.ckpt", "--repo", "org/model"])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output
