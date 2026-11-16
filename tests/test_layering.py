"""Package dependency direction and import cost."""

import subprocess
import sys

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


@pytest.mark.slow
def test_geodata_loads_no_torch():
    assert not _loads(
        "geosave.geodata, geosave.geodata.io",
        "torch",
    )


@pytest.mark.slow
def test_model_core_loads_no_lightning():
    imports = (
        "geosave.ml.chain, geosave.ml.components.build, geosave.ml.components.head, "
        "geosave.ml.processing.raster.pipeline"
    )
    assert not _loads(imports, "lightning")


@pytest.mark.slow
def test_ml_package_loads_no_torch():
    assert not _loads("geosave.ml", "torch")


@pytest.mark.slow
def test_library_loads_no_flows_or_prefect():
    imports = "geosave, geosave.ml, geosave.cli.main"
    loaded = _run(
        f"import sys\nimport {imports}\n"
        "print([m for m in ('geosave_flows', 'prefect') if m in sys.modules])"
    )
    assert loaded == "[]"


@pytest.mark.slow
def test_release_loads_no_lightning():
    assert not _loads("geosave.ml.release", "lightning")
