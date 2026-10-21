"""Discover tasks and methods without offering cache directories."""

from geosave.cli.core import templates


def test_template_discovery_ignores_caches(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    for task, method in [
        ("segmentation", "supervised"),
        ("segmentation", "fixmatch"),
        ("lightning", "boilerplate"),
    ]:
        (tasks / task / method).mkdir(parents=True)
    (tasks / "__pycache__/cache").mkdir(parents=True)
    (tasks / "segmentation/.ipynb_checkpoints").mkdir()
    (tasks / "empty").mkdir()
    monkeypatch.setattr(templates, "TASKS_DIR", tasks)
    assert templates.get_tasks() == ["lightning", "segmentation"]
    assert templates.get_methods("segmentation") == ["supervised", "fixmatch"]
    assert templates.get_methods("unknown") == []


def test_source_templates_are_siblings_of_library_packages():
    from pathlib import Path

    root = Path(__file__).parents[3]
    assert templates.TEMPLATES_DIR == root / "src/templates"
    assert (templates.COMMON_DIR / "main.py").is_file()
    assert templates.describe("segmentation", "supervised")
