"""Package dependency direction and registration behavior."""

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
    root = Path(__file__).parents[1] / "src" / "geosave_engine" / "geodata"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        assert not _names(_imports(ast.walk(tree)), "geosave_engine.ml"), path
        # `.gs.to_tensor()` imports torch when called; importing geodata must not.
        assert not _names(_imports(_import_time_nodes(tree)), "torch"), path


@pytest.mark.slow
def test_geodata_loads_no_torch():
    assert not _loads(
        "geosave_engine.geodata, geosave_engine.geodata.io, geosave_engine.geodata.benchmarks",
        "torch",
    )


@pytest.mark.slow
def test_model_core_loads_no_lightning():
    imports = "geosave_engine.model.chain, geosave_engine.model.registry, geosave_engine.model.head"
    assert not _loads(imports, "lightning")


@pytest.mark.slow
def test_every_stage_package_registers_its_models():
    listed = _run(
        "from geosave_engine.model.registry import list_models\nprint(list_models())"
    )
    assert ast.literal_eval(listed) == {
        "decoder": ["DPT", "UNET"],
        "encoder": ["CLAY", "DINOV3", "PRITHVI", "PRITHVI_TL"],
        "head": ["CLASSIFICATION", "DETECTION", "REGRESSION", "SEGMENTATION"],
        "model": ["IBM_GRANITE_BIOMASS"],
    }


@pytest.mark.slow
def test_spec_and_workflow_load_no_torch():
    assert not _loads(
        "geosave_engine.model.spec, geosave_engine.workflow.flows", "torch"
    )


@pytest.mark.slow
def test_release_loads_no_lightning():
    assert not _loads("geosave_engine.model.release", "lightning")


def test_supervised_segmentation_pairs_its_module_and_data():
    from lightning import LightningDataModule, LightningModule

    from geosave_engine.ml.segmentation import supervised

    assert issubclass(supervised.Module, LightningModule)
    assert issubclass(supervised.DataModule, LightningDataModule)
