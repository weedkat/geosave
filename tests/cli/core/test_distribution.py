"""Installed distributions include the sibling workspace resources."""

import os
from pathlib import Path
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest


@pytest.mark.slow
def test_built_wheel_generates_every_recipe_outside_source_tree(tmp_path):
    root = Path(__file__).parents[3]
    build = subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(tmp_path / "dist")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr
    source = next((tmp_path / "dist").glob("*.tar.gz"))
    with tarfile.open(source) as archive:
        assert any(
            name.endswith("/src/templates/common/main.py")
            for name in archive.getnames()
        )
    wheel = next((tmp_path / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "templates/common/main.py" in names
        assert "templates/common/.gitignore" in names
        assert "templates/common/.env.example" in names
        assert not any(name.startswith("geosave/templates/") for name in names)
    install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--python",
            sys.executable,
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert install.returncode == 0, install.stderr

    script = """
import os
from pathlib import Path
import geosave
from typer.testing import CliRunner
from geosave.cli.main import app
from geosave.cli.core.templates import TEMPLATES_DIR, get_tasks, get_methods

installed = Path('installed').resolve()
assert Path(geosave.__file__).is_relative_to(installed)
assert TEMPLATES_DIR == installed / 'templates'
runner = CliRunner()
result = runner.invoke(app, ['create', 'demo', '--task', 'blank'])
assert result.exit_code == 0, result.output
os.chdir('demo')
assert Path('.gitignore').is_file()
assert Path('.env.example').is_file()
count = 0
for task in get_tasks():
    for method in get_methods(task):
        result = runner.invoke(app, ['add-task', '--task', task, '--method', method])
        assert result.exit_code == 0, result.output
        assert Path(f'modules/{task}/{method.replace("-", "_")}/module.py').is_file()
        count += 1
assert count == 7
assert len(list(Path('configs').glob('*.yaml'))) == 7
"""
    env = dict(os.environ, PYTHONPATH=str(installed))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
