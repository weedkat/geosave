"""Execute the public scenario matrix against real saved rasters."""

import runpy
from pathlib import Path


def test_public_sample_frame_scenarios():
    root = Path(__file__).parents[3]
    runpy.run_path(
        str(
            root
            / "docs/contributing/handoffs/2026-10-16-sample-frame/probe_sample_api.py"
        )
    )
