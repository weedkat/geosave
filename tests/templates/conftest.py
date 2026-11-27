"""Import each starter from its own generated workspace for the recipe unit tests."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from geosave.cli.core.templates import get_methods, get_tasks
from geosave.cli.core.workspace import create_workspace


def pytest_configure(config):
    folder = TemporaryDirectory(prefix="geosave-training-kits-")
    root = Path(folder.name)
    for task in get_tasks():
        for method in get_methods(task):
            project = root / "generated_kits" / task / method.replace("-", "_")
            create_workspace(project, task, method)
    sys.path.insert(0, str(root))
    config.add_cleanup(lambda: sys.path.remove(str(root)))
    config.add_cleanup(folder.cleanup)
