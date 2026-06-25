"""quant-wheel dashboard (Streamlit).

Run with::

    streamlit run src/quantwheel/ui/streamlit_app/app.py

This file is a *thin* adapter: it lays out tabs and a sidebar and delegates every
computation to the ``services`` facade. It imports no model code directly — which
is exactly why it could be swapped for the Textual TUI (``ui/tui``) without
touching the backend.
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from quantwheel.core.config import DEFAULT
from quantwheel.ui.streamlit_app.tabs import (
    approve,
    backtest,
    candidate_detail,
    council,
    scan,
    tracker,
    universe,
)

st.set_page_config(page_title="quant-wheel", page_icon="🎯", layout="wide")


def _sidebar_config():
    """Account parameters live in the sidebar and flow into every service call."""
    st.sidebar.title("🎯 quant-wheel")
    st.sidebar.caption("Small-account Wheel engine · educational, not advice")
    st.sidebar.subheader("Account")
    account = st.sidebar.number_input("Account ($)", 100.0, 100_000.0,
                                      float(DEFAULT.account), step=100.0)
    buffer = st.sidebar.number_input("Cash buffer ($)", 0.0, 50_000.0,
                                     float(DEFAULT.cash_buffer), step=50.0)
    st.sidebar.metric("Deployable", f"${max(0.0, account - buffer):,.0f}")
    st.sidebar.divider()
    st.sidebar.caption(
        "Flow: **Universe → Scan → Detail → Council → Approve → Tracker**, "
        "plus **Backtest** for the evidence. No order is ever placed by software."
    )
    return replace(DEFAULT, account=account, cash_buffer=buffer)


def main() -> None:
    config = _sidebar_config()
    st.session_state["config"] = config

    tabs = st.tabs([
        "🗂 Universe", "🔎 Scan", "🔬 Candidate", "🏛 Council",
        "✅ Approve", "📒 Tracker", "📈 Backtest",
    ])
    with tabs[0]:
        universe.render()
    with tabs[1]:
        scan.render(config)
    with tabs[2]:
        candidate_detail.render()
    with tabs[3]:
        council.render()
    with tabs[4]:
        approve.render(config)
    with tabs[5]:
        tracker.render()
    with tabs[6]:
        backtest.render(config)


if __name__ == "__main__":
    main()
