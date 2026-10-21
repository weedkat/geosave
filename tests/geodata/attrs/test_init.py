from __future__ import annotations

import subprocess
import sys

import geosave.geodata  # noqa: F401  — registers the .gs accessor


def test_importing_attrs_does_not_import_stac() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import geosave.geodata.attrs; "
            "assert 'geosave.geodata.stac' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
