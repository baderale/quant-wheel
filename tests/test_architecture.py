"""Architecture guard: the backend must never import a UI library.

This is the executable form of the project's #1 rule (``ui -> services -> core``,
one-way). If someone sneaks ``import streamlit`` into a model, this test fails —
so swapping the UI stays a self-contained change forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "quantwheel"
_BACKEND_DIRS = ["core", "services", "agents"]
_FORBIDDEN = ("streamlit", "textual", "plotly", "dash", "gradio")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for d in _BACKEND_DIRS:
        files.extend((_SRC / d).rglob("*.py"))
    return files


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_ui_imports_in_backend(path: Path):
    text = path.read_text(encoding="utf-8")
    for lib in _FORBIDDEN:
        assert f"import {lib}" not in text, (
            f"{path.relative_to(_SRC)} imports UI library {lib!r} — "
            f"backend layers must stay UI-agnostic."
        )
