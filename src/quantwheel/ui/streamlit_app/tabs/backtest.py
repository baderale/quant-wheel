"""Backtest tab — the evidence: equity curve, metrics, ablation, calibration."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from quantwheel.services.backtest_service import run_backtest
from quantwheel.universe import DEFAULT_UNIVERSE


def render(config) -> None:
    st.subheader("Backtest & rule ablation")
    st.caption("Walk-forward wheel-cycle simulation. The ablation shows how much "
               "each rule contributes. Premiums are modelled on realized vol "
               "(no historical IV) — see the notebook's limitation note.")

    universe = st.session_state.get("universe", DEFAULT_UNIVERSE)
    tickers = st.multiselect("Tickers", universe, default=universe[:4])
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", date(2024, 1, 1))
    end = c2.date_input("End", date(2026, 6, 1))

    if st.button("▶ Run backtest", type="primary") and tickers:
        with st.spinner("Simulating the wheel through history (incl. ablation runs)…"):
            st.session_state["backtest"] = run_backtest(tickers, start, end, config)

    res = st.session_state.get("backtest")
    if res is None:
        st.info("Run a backtest to see results.")
        return

    eq = pd.Series({pd.Timestamp(d): v for d, v in res.equity_curve}).sort_index()
    st.line_chart(eq, height=260, use_container_width=True)

    m = res.metrics
    cols = st.columns(5)
    cols[0].metric("CAGR", f"{m['cagr']:.1%}")
    cols[1].metric("Sharpe", f"{m['sharpe']:.2f}")
    cols[2].metric("Max DD", f"{m['max_drawdown']:.1%}")
    cols[3].metric("Win rate", f"{m['win_rate']:.0%}")
    cols[4].metric("Trades", res.n_trades)

    st.markdown("**Rule ablation** (each rule turned OFF — lower Sharpe = the rule helps)")
    rows = {"baseline (all on)": res.metrics}
    rows.update(res.ablation)
    tbl = pd.DataFrame(rows).T[["sharpe", "cagr", "max_drawdown", "win_rate"]]
    st.dataframe(tbl.style.format({
        "sharpe": "{:.2f}", "cagr": "{:.1%}",
        "max_drawdown": "{:.1%}", "win_rate": "{:.0%}",
    }), use_container_width=True)

    if res.calibration:
        st.markdown("**Calibration** — predicted P(assign) vs realized frequency "
                    "(diagonal = perfect; below = conservative)")
        cal = pd.DataFrame(res.calibration, columns=["predicted", "realized"])
        st.scatter_chart(cal, x="predicted", y="realized", height=300)
