"""Only task templates are selectable."""


def test_template_discovery_ignores_caches(tmp_path, monkeypatch):
    from geosave.cli.core import templates

    tasks = tmp_path / "tasks"
    for name in [
        "segmentation-supervised",
        "lightning-boilerplate",
        "__pycache__",
        ".ipynb_checkpoints",
    ]:
        (tasks / name).mkdir(parents=True)
    monkeypatch.setattr(templates, "TASKS_DIR", tasks)
    assert templates.get_tasks() == ["lightning-boilerplate", "segmentation-supervised"]
