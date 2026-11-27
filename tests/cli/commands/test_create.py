"""Generate independent, editable training workspaces through the public CLI."""

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


def test_blank_has_only_shared_startup(tmp_path):
    root = tmp_path / "demo"
    result = CliRunner().invoke(app, ["create", str(root), "--task", "blank"])
    assert result.exit_code == 0, result.output
    assert (root / "main.py").is_file()
    assert (root / ".env.example").is_file()
    assert not (root / "training").exists()
    assert not (root / "configs").exists()
    assert not (root / "artifacts").exists()


@pytest.mark.parametrize(("task", "method"), RECIPES)
def test_create_with_explicit_recipe(tmp_path, task, method):
    root = tmp_path / "demo"
    result = CliRunner().invoke(app, ["create", str(root), "-t", task, "-m", method])
    assert result.exit_code == 0, result.output
    assert (root / "training" / "__init__.py").is_file()
    assert (root / "configs" / f"{method}.yaml").is_file()
    assert (root / "README.md").is_file()
    if task != "lightning":
        assert (root / "pipeline.py").is_file()
        assert (root / "processing").is_dir()
    assert not (root / "description.txt").exists()
    assert not (root / "project.yaml").exists()


def test_create_prompts_task_then_method(tmp_path, monkeypatch):
    answers = iter(["segmentation", "fixmatch"])
    prompts = []

    def select(message, choices):
        prompts.append(message)
        if "method" in message.lower():
            assert [choice.value for choice in choices] == ["supervised", "fixmatch"]
        return next(answers)

    monkeypatch.setattr("geosave.cli.core.selection.prompt_select", select)
    root = tmp_path / "demo"
    result = CliRunner().invoke(app, ["create", str(root)])
    assert result.exit_code == 0, result.output
    assert len(prompts) == 2
    assert (root / "configs/fixmatch.yaml").is_file()


def test_only_one_method_needs_no_second_prompt(tmp_path):
    root = tmp_path / "demo"
    result = CliRunner().invoke(app, ["create", str(root), "--task", "lightning"])
    assert result.exit_code == 0, result.output
    assert (root / "configs/boilerplate.yaml").is_file()


@pytest.mark.parametrize(
    "options",
    [
        ["-t", "segmentation", "-m", "mean-teacher"],
        ["-t", "unknown"],
        ["-t", "blank", "-m", "supervised"],
    ],
)
def test_invalid_selection_creates_nothing(tmp_path, options):
    root = tmp_path / "demo"
    result = CliRunner().invoke(app, ["create", str(root), *options])
    assert result.exit_code == 2
    assert not root.exists()


def test_force_permits_unrelated_files_without_reading_project_metadata(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "project.yaml").write_text("user: [metadata]\n")
    runner = CliRunner()
    args = ["create", str(root), "-t", "blank"]
    assert runner.invoke(app, args).exit_code == 2
    assert not (root / "main.py").exists()
    result = runner.invoke(app, [*args, "--force"])
    assert result.exit_code == 0, result.output
    assert (root / "project.yaml").read_text() == "user: [metadata]\n"


def test_force_reports_all_conflicts_before_copying(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / "main.py").write_text("user code")
    (root / "training").write_text("user file")
    result = CliRunner().invoke(
        app, ["create", str(root), "-t", "lightning", "--force"]
    )
    assert result.exit_code == 2
    assert "main.py" in result.output
    assert "training" in result.output
    assert sorted(p.name for p in root.iterdir()) == ["main.py", "training"]
    assert (root / "main.py").read_text() == "user code"


def test_force_cannot_write_through_an_existing_directory_link(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    external = tmp_path / "elsewhere"
    external.mkdir()
    (root / "training").symlink_to(external, target_is_directory=True)
    result = CliRunner().invoke(
        app, ["create", str(root), "-t", "lightning", "--force"]
    )
    assert result.exit_code == 2
    assert "training" in result.output
    assert not list(external.iterdir())
    assert not (root / "main.py").exists()
