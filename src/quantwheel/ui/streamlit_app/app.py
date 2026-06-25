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
    report,
    scan,
    tracker,
    universe,
)

st.set_page_config(page_title="quant-wheel", page_icon="🎯", layout="wide")

# Session keys holding *computed* results. They depend on the watchlist + account
# settings, so they're cleared whenever those inputs change (or on explicit reset).
_RESULT_KEYS = ("scan", "council", "report", "backtest", "discovery")


def _sidebar_config():
    """Account parameters live in the sidebar and flow into every service call."""
    st.sidebar.title("🎯 quant-wheel")
    st.sidebar.caption("Account-aware Wheel engine · educational, not advice")
    st.sidebar.subheader("Account")
    account = st.sidebar.number_input("Account ($)", 100.0, 1_000_000.0,
                                      float(DEFAULT.account), step=100.0)
    buffer = st.sidebar.number_input("Cash buffer ($)", 0.0, 500_000.0,
                                     float(DEFAULT.cash_buffer), step=50.0)
    max_positions = int(st.sidebar.number_input(
        "Max concurrent positions", 1, 20, int(DEFAULT.max_positions), step=1))

    deployable = max(0.0, account - buffer)
    st.sidebar.metric("Deployable", f"${deployable:,.0f}")
    if max_positions > 1:
        st.sidebar.caption(f"≈ ${deployable / max_positions:,.0f} budget per position, "
                           f"across up to {max_positions} names.")
    return replace(DEFAULT, account=account, cash_buffer=buffer,
                   max_positions=max_positions)


def _invalidate_stale_results(config) -> None:
    """Clear computed results when the watchlist or account settings change.

    This is what makes 'go back, add more tickers, re-run' behave: stale scan /
    council / report / backtest are dropped so every dependent tab prompts a
    fresh run instead of silently showing yesterday's numbers.
    """
    universe = st.session_state.get("universe")
    sig = (tuple(universe) if universe else None,
           config.account, config.cash_buffer, config.max_positions)
    if st.session_state.get("_inputs_sig") != sig:
        if st.session_state.get("_inputs_sig") is not None:  # not the first load
            for key in _RESULT_KEYS:
                st.session_state.pop(key, None)
        st.session_state["_inputs_sig"] = sig


def main() -> None:
    config = _sidebar_config()
    st.session_state["config"] = config
    _invalidate_stale_results(config)

    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset results"):
        for key in _RESULT_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.caption(
        "Flow: **Universe → Scan → Detail → Council → Report → Approve → "
        "Tracker**, plus **Backtest** for the evidence. No order is ever placed."
    )

    tabs = st.tabs([
        "🗂 Universe", "🔎 Scan", "🔬 Candidate", "🏛 Council",
        "📋 Report", "✅ Approve", "📒 Tracker", "📈 Backtest",
    ])
    with tabs[0]:
        universe.render(config)
    with tabs[1]:
        scan.render(config)
    with tabs[2]:
        candidate_detail.render()
    with tabs[3]:
        council.render(config)
    with tabs[4]:
        report.render(config)
    with tabs[5]:
        approve.render(config)
    with tabs[6]:
        tracker.render()
    with tabs[7]:
        backtest.render(config)


if __name__ == "__main__":
    main()
