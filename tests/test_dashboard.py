"""Dashboard smoke test — runs the Streamlit script headlessly and asserts it
renders without raising. Initial render presses no buttons, so it needs neither
network nor an API key."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

_APP = Path(__file__).resolve().parents[1] / "src" / "quantwheel" / "ui" / "streamlit_app" / "app.py"


def test_dashboard_renders_without_exception():
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    assert not at.exception, f"dashboard raised: {at.exception}"
    # Sanity: the eight tabs are present (incl. the new Report tab).
    assert len(at.tabs) == 8
