"""Package dependency direction and import cost."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


def _run(code: str) -> str:
    """Return the last line a fresh interpreter prints for `code`."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip().splitlines()[-1]


def _loads(imports: str, module: str) -> bool:
    """Report whether importing `imports` also loads `module`."""
    return (
        _run(f"import sys\nimport {imports}\nprint({module!r} in sys.modules)")
        == "True"
    )


def _imports(nodes) -> list[str]:
    """List the modules the given nodes import."""
    modules = []
    for node in nodes:
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


_TYPE_CHECKING = {"TYPE_CHECKING", "typing.TYPE_CHECKING"}


def _import_time_nodes(tree: ast.Module):
    """Yield the nodes a module runs when imported.

    Function bodies run when called and `TYPE_CHECKING` blocks never run, so
    neither is entered.
    """
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) in _TYPE_CHECKING:
            pending.extend(node.orelse)
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def _names(modules: list[str], package: str) -> bool:
    return any(m == package or m.startswith(f"{package}.") for m in modules)


def test_import_time_nodes_skip_what_importing_does_not_run():
    tree = ast.parse(
        "import a\n"
        "if TYPE_CHECKING:\n    import b\nelse:\n    import c\n"
        "if not TYPE_CHECKING:\n    import h\n"
        "def f():\n    import d\n"
        "class K:\n    import e\n    def m(self):\n        import g\n"
    )

    assert sorted(_imports(_import_time_nodes(tree))) == ["a", "c", "e", "h"]


def test_geodata_imports_no_ml_and_no_torch_at_import_time():
    root = Path(__file__).parents[1] / "src" / "geosave" / "geodata"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        assert not _names(_imports(ast.walk(tree)), "geosave.ml"), path
        # `.gs.to_tensor()` imports torch when called; importing geodata must not.
        assert not _names(_imports(_import_time_nodes(tree)), "torch"), path


def test_only_ml_tasks_import_ml_tasks():
    root = Path(__file__).parents[1] / "src" / "geosave" / "ml"
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "tasks":
            continue
        tree = ast.parse(path.read_text())
        assert not _names(_imports(ast.walk(tree)), "geosave.ml.tasks"), path


@pytest.mark.slow
def test_geodata_loads_no_torch():
    assert not _loads(
        "geosave.geodata, geosave.geodata.io",
        "torch",
    )


@pytest.mark.slow
def test_model_core_loads_no_lightning():
    imports = "geosave.model.chain, geosave.model.registry, geosave.model.stages.head"
    assert not _loads(imports, "lightning")


@pytest.mark.slow
def test_spec_loads_no_torch():
    assert not _loads("geosave.model.spec", "torch")


@pytest.mark.slow
def test_library_loads_no_flows_or_prefect():
    imports = "geosave, geosave.model.spec, geosave.ml, geosave.cli.main"
    loaded = _run(
        f"import sys\nimport {imports}\n"
        "print([m for m in ('geosave_flows', 'prefect') if m in sys.modules])"
    )
    assert loaded == "[]"


@pytest.mark.slow
def test_release_loads_no_lightning():
    assert not _loads("geosave.model.release", "lightning")
