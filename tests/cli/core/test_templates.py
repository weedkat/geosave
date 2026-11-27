"""Discover only methods with catalog metadata."""

from geosave.cli.core import templates


def test_template_discovery_ignores_non_recipes(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    for task, method in [
        ("segmentation", "supervised"),
        ("segmentation", "fixmatch"),
        ("lightning", "boilerplate"),
    ]:
        folder = tasks / task / "methods" / method
        folder.mkdir(parents=True)
        (folder / "description.txt").write_text(f"{method} starter")
    (tasks / "segmentation/common").mkdir()
    (tasks / "segmentation/methods/__pycache__").mkdir()
    (tasks / "empty").mkdir()
    monkeypatch.setattr(templates, "TASKS_DIR", tasks)
    assert templates.get_tasks() == ["lightning", "segmentation"]
    assert templates.get_methods("segmentation") == ["supervised", "fixmatch"]
    assert templates.get_methods("unknown") == []
    assert (
        templates.read_method_description("segmentation", "supervised")
        == "supervised starter"
    )
