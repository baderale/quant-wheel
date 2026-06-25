"""A minimal Textual TUI — the *proof* of the decoupling rule.

This terminal UI renders the weekly scan in a few dozen lines, calling the exact
same ``services`` facade the Streamlit dashboard uses. It imports **zero** model
code. That's the whole point: changing "the UI to a TUI" was a self-contained
addition under ``ui/`` — the math, council, and agent were never touched.

Run with::

    python -m quantwheel.ui.tui.app

Keys: ``s`` run the scan · ``q`` quit.
"""

from __future__ import annotations

from dataclasses import replace

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from quantwheel.core.config import DEFAULT
from quantwheel.services.scan_service import run_weekly_scan  # the SAME facade
from quantwheel.universe import DEFAULT_UNIVERSE

_COLUMNS = ("Ticker", "Verdict", "Score", "Strike", "DTE", "Credit$", "POP",
            "EV$", "P(assign)", "Fits")


class WheelTUI(App):
    """Same backend, a terminal face."""

    TITLE = "quant-wheel (TUI)"
    BINDINGS = [("s", "scan", "Run scan"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"Universe: {', '.join(DEFAULT_UNIVERSE)}  —  press [b]s[/b] to scan",
                     id="status")
        yield DataTable(id="results")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns(*_COLUMNS)

    def action_scan(self) -> None:
        self.query_one("#status", Static).update("Scanning (live data)…")
        config = replace(DEFAULT, account=DEFAULT.account)
        scan = run_weekly_scan(DEFAULT_UNIVERSE, config)
        table = self.query_one(DataTable)
        table.clear()
        for c in scan.candidates:
            q, p = c.quote, c.pricing
            table.add_row(
                c.ticker, c.verdict.value, f"{c.score:.1f}", f"${q.strike:.2f}",
                str(q.dte), f"${p.credit:.0f}", f"{p.pop:.0%}",
                f"${p.expected_value:.1f}", f"{c.monte_carlo.p_assign:.0%}",
                str(c.sizing.contracts),
            )
        self.query_one("#status", Static).update(
            f"{len(scan.enterable)} enterable / {len(scan.candidates)} scored")


def main() -> None:
    WheelTUI().run()


if __name__ == "__main__":
    main()
