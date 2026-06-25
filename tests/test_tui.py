"""TUI smoke test — mounts the Textual app headlessly (no scan, so no network).

Proves the alternative UI composes and binds to the same services without error.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")
from quantwheel.ui.tui.app import WheelTUI, _COLUMNS


def test_tui_mounts_with_results_table():
    async def _go():
        app = WheelTUI()
        async with app.run_test() as pilot:  # headless pilot
            await pilot.pause()
            from textual.widgets import DataTable
            table = app.query_one("#results", DataTable)
            assert len(table.columns) == len(_COLUMNS)

    asyncio.run(_go())
