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
    loaded = _run(
        """
import sys
from geosave.ml.torch import ModelChain

ModelChain.from_config({
    "head": {"name": "segmentation", "init_args": {"feature_channels": 4, "num_classes": 2}},
})
print("lightning" in sys.modules)
"""
    )
    assert loaded == "False"


@pytest.mark.slow
def test_framework_free_ml_loads_no_torch():
    assert not _loads("geosave.ml, geosave.ml.torch", "torch")


@pytest.mark.slow
def test_library_loads_no_flows_or_prefect():
    imports = "geosave, geosave.ml, geosave.cli.main"
    loaded = _run(
        f"import sys\nimport {imports}\n"
        "print([m for m in ('geosave_flows', 'prefect') if m in sys.modules])"
    )
    assert loaded == "[]"


@pytest.mark.slow
def test_the_chain_loads_no_lightning():
    assert not _loads("geosave.ml.torch.chain", "lightning")
