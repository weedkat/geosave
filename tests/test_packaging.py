"""Declared dependencies match what each package imports."""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
SOURCES = [
    path
    for path in (ROOT / "src").rglob("*.py")
    if "templates" not in path.relative_to(ROOT / "src").parts
]

# Module-level imports of each package resolve from core plus these extras.
# The longest matching prefix wins.
AREAS: dict[str, set[str]] = {
    "geosave": set(),
    "geosave/model": {"ml", "hub"},
    "geosave/model/spec": set(),
    "geosave/ml": {"ml"},
    "geosave_flows": {"flows"},
}
# Installed for a runtime role, never imported by name.
RUNTIME_ONLY = {
    "jupyter-bokeh",  # panel widgets in notebooks
    "netcdf4",  # xarray's default netCDF engine
    "tensorboard",  # Lightning's TensorBoardLogger, the CLI default
    "mlflow",  # Lightning's MLFlowLogger, added when MLFLOW_TRACKING_URI is set
    "s3fs",  # fsspec s3:// protocol
    "gcsfs",  # fsspec gs:// protocol
}
FIRST_PARTY = {"geosave", "geosave_flows", "__future__"}
DISTRIBUTIONS = packages_distributions()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    return _normalize(re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0])


def _declared(extras: set[str]) -> set[str]:
    names = {_requirement_name(item) for item in PROJECT["dependencies"]}
    for extra in extras:
        names |= {
            _requirement_name(item)
            for item in PROJECT["optional-dependencies"][extra]
        }
    return names - {"geosave"}


def _third_party(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        modules = [node.module]
    else:
        return []
    tops = [module.split(".")[0] for module in modules]
    return [
        top for top in tops if top not in FIRST_PARTY and top not in sys.stdlib_module_names
    ]


def _module_level(tree: ast.Module) -> list[str]:
    """Imports a module runs on import; TYPE_CHECKING blocks never run."""
    found = []
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            continue
        found += _third_party(node)
    return found


def _area(path: Path) -> str:
    package = path.relative_to(ROOT / "src").as_posix()
    return max((prefix for prefix in AREAS if package.startswith(prefix)), key=len)


def _distributions(module: str) -> set[str]:
    return {_normalize(name) for name in DISTRIBUTIONS.get(module, [module])}


def test_module_level_imports_are_declared_for_their_package():
    missing = {}
    for path in SOURCES:
        area = _area(path)
        allowed = _declared(AREAS[area])
        for module in _module_level(ast.parse(path.read_text())):
            if not _distributions(module) & allowed:
                missing.setdefault(module, set()).add(area)
    assert missing == {}


def test_every_declared_dependency_is_imported_or_runtime_only():
    imported = set()
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.stmt):
                for module in _third_party(node):
                    imported |= _distributions(module)
    declared = _declared(set(PROJECT["optional-dependencies"]))
    assert declared - imported - RUNTIME_ONLY == set()


def test_all_extra_installs_every_other_extra():
    extras = set(PROJECT["optional-dependencies"]) - {"all"}
    (requirement,) = PROJECT["optional-dependencies"]["all"]
    assert requirement.startswith("geosave[")
    assert set(requirement[len("geosave[") : -1].split(",")) == extras
