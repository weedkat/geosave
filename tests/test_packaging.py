"""Installed artifacts support the advertised optional features and workspaces."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_all_extra_installs_every_other_extra():
    extras = set(PROJECT["optional-dependencies"]) - {"all"}
    (requirement,) = PROJECT["optional-dependencies"]["all"]
    assert requirement.startswith("geosave[")
    assert set(requirement[len("geosave[") : -1].split(",")) == extras


@pytest.mark.slow
def test_built_wheel_generates_every_recipe_outside_source_tree(tmp_path):
    root = ROOT
    build = subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(tmp_path / "dist")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
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
from geosave.cli.core.templates import get_tasks, get_methods

installed = Path('installed').resolve()
assert Path(geosave.__file__).is_relative_to(installed)
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
assert count > 0
assert len(list(Path('configs').glob('*.yaml'))) == count
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
