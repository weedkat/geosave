"""Workspace task and method selection through the public CLI."""

import pytest
from typer.testing import CliRunner

from geosave.cli.main import app

RECIPES = [
    ("segmentation", "supervised"),
    ("segmentation", "fixmatch"),
    ("regression", "supervised"),
    ("regression", "mean-teacher"),
    ("classification", "supervised"),
    ("pretraining", "mim"),
    ("lightning", "boilerplate"),
]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["create", "demo", "--task", "blank"])
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(tmp_path / "demo")
    return tmp_path / "demo"


def test_blank_has_only_shared_startup(workspace):
    assert (workspace / "main.py").is_file()
    assert (workspace / ".env.example").is_file()
    assert not any((workspace / "configs").iterdir())


@pytest.mark.parametrize(("task", "method"), RECIPES)
def test_create_with_explicit_recipe(tmp_path, monkeypatch, task, method):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["create", "demo", "-t", task, "-m", method])
    assert result.exit_code == 0, result.output
    root = tmp_path / "demo"
    package = root / "modules" / task / method.replace("-", "_")
    for name in ("module.py", "data.py", "datasets.py"):
        assert (package / name).is_file()
    assert (root / f"configs/{task}-{method}.yaml").is_file()


def test_create_prompts_task_then_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["segmentation", "fixmatch"])
    prompts = []

    def select(message, choices):
        prompts.append(message)
        return next(answers)

    monkeypatch.setattr("geosave.cli.core.selection.prompt_select", select)
    result = CliRunner().invoke(app, ["create", "demo"])
    assert result.exit_code == 0, result.output
    assert len(prompts) == 2
    assert "task" in prompts[0].lower()
    assert "method" in prompts[1].lower()
    assert (tmp_path / "demo/modules/segmentation/fixmatch/module.py").is_file()


def test_create_prompts_only_missing_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prompts = []

    def select(message, choices):
        prompts.append(message)
        return "supervised"

    monkeypatch.setattr("geosave.cli.core.selection.prompt_select", select)
    result = CliRunner().invoke(app, ["create", "demo", "-t", "regression"])
    assert result.exit_code == 0, result.output
    assert len(prompts) == 1
    assert "method" in prompts[0].lower()


@pytest.mark.parametrize(
    "options",
    [
        ["-t", "missing", "-m", "supervised"],
        ["-t", "segmentation", "-m", "mean-teacher"],
        ["-t", "blank", "-m", "supervised"],
    ],
)
def test_invalid_selection_creates_nothing(tmp_path, monkeypatch, options):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["create", "demo", *options])
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "demo").exists()


def test_nonempty_directory_requires_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "demo"
    root.mkdir()
    (root / "notes.txt").write_text("keep")
    runner = CliRunner()
    refused = runner.invoke(app, ["create", "demo", "-t", "blank"])
    assert refused.exit_code == 2
    assert not (root / "main.py").exists()
    forced = runner.invoke(app, ["create", "demo", "-t", "blank", "--force"])
    assert forced.exit_code == 0, forced.output
    assert (root / "notes.txt").read_text() == "keep"


@pytest.mark.parametrize(("task", "method"), RECIPES)
def test_add_recipe(workspace, task, method):
    result = CliRunner().invoke(app, ["add-task", "-t", task, "-m", method])
    assert result.exit_code == 0, result.output
    assert (workspace / f"configs/{task}-{method}.yaml").is_file()


def test_add_preserves_existing_implementation_and_startup(workspace):
    runner = CliRunner()
    result = runner.invoke(app, ["add-task", "-t", "segmentation", "-m", "supervised"])
    assert result.exit_code == 0, result.output
    module = workspace / "modules/segmentation/supervised/module.py"
    module.write_text("user_code = True\n")
    (workspace / "main.py").write_text("user_startup = True\n")
    result = runner.invoke(app, ["add-task", "-t", "segmentation", "-m", "fixmatch"])
    assert result.exit_code == 0, result.output
    assert module.read_text() == "user_code = True\n"
    assert (workspace / "main.py").read_text() == "user_startup = True\n"
    assert (workspace / "configs/segmentation-supervised.yaml").is_file()
    assert (workspace / "configs/segmentation-fixmatch.yaml").is_file()


def test_duplicate_recipe_leaves_files_untouched(workspace):
    runner = CliRunner()
    options = ["add-task", "-t", "segmentation", "-m", "supervised"]
    assert runner.invoke(app, options).exit_code == 0
    module = workspace / "modules/segmentation/supervised/module.py"
    module.write_text("mine = True\n")
    result = runner.invoke(app, options)
    assert result.exit_code == 2
    assert "already exists" in result.output
    assert module.read_text() == "mine = True\n"


def test_add_requires_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["add-task", "-t", "segmentation", "-m", "supervised"]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "configs").exists()


def test_publish_reports_not_implemented():
    result = CliRunner().invoke(app, ["publish", "run.ckpt", "--repo", "org/model"])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_add_selector_disables_existing_method(workspace, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(app, ["add-task", "-t", "segmentation", "-m", "supervised"])
    assert result.exit_code == 0, result.output
    shown = {}

    def select(message, choices):
        for choice in choices:
            shown[choice.value] = choice.disabled
        return "fixmatch"

    monkeypatch.setattr("geosave.cli.core.selection.prompt_select", select)
    result = runner.invoke(app, ["add-task", "-t", "segmentation"])
    assert result.exit_code == 0, result.output
    assert shown["supervised"] == "already exists"
    assert not shown["fixmatch"]


def test_add_reports_when_every_method_exists(workspace, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(
        app, ["add-task", "-t", "classification", "-m", "supervised"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["add-task", "-t", "classification"])
    assert result.exit_code == 2
    assert "already exist" in result.output
