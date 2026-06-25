"""Tracker tab — the built-in trade journal (open/closed cycles, P&L)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from quantwheel.core.types import TradeRecord
from quantwheel.services import tracker_service as ts


def render() -> None:
    st.subheader("Trade tracker")

    s = ts.summary()
    cols = st.columns(4)
    cols[0].metric("Open positions", int(s["n_open"]))
    cols[1].metric("Realized P&L", f"${s['realized_pnl']:,.0f}")
    cols[2].metric("Win rate", f"{s['win_rate']:.0%}")
    cols[3].metric("Open collateral", f"${s['open_collateral']:,.0f}")

    trades = ts.list_trades()
    if trades:
        st.dataframe(pd.DataFrame([{
            "id": t.id, "ticker": t.ticker, "leg": t.leg, "phase": t.phase,
            "status": t.status, "opened": t.open_date, "strike": t.strike,
            "expiry": t.expiry, "qty": t.contracts, "credit": t.credit_open,
            "pnl": t.realized_pnl,
        } for t in trades]), hide_index=True, use_container_width=True)
    else:
        st.info("No trades yet. Approve one from the Approve tab, or add manually below.")

    col_close, col_add = st.columns(2)

    with col_close:
        st.markdown("**Close a trade**")
        open_ids = [t.id for t in trades if t.status == "OPEN"]
        if open_ids:
            tid = st.selectbox("Trade id", open_ids)
            debit = st.number_input("Buy-to-close debit ($)", 0.0, value=0.0, step=1.0)
            pnl = st.number_input("Realized P&L ($)", value=0.0, step=1.0)
            if st.button("Close trade"):
                ts.close_trade(tid, debit, pnl)
                st.success(f"Closed {tid}.")
                st.rerun()
        else:
            st.caption("No open trades.")

    with col_add:
        st.markdown("**Add a trade manually**")
        with st.form("add_trade", clear_on_submit=True):
            tk = st.text_input("Ticker").upper()
            strike = st.number_input("Strike", 0.0, value=5.0, step=0.5)
            credit = st.number_input("Credit collected ($)", 0.0, value=10.0, step=1.0)
            qty = st.number_input("Contracts", 1, value=1)
            if st.form_submit_button("Add") and tk:
                ts.add_trade(TradeRecord(
                    id="", ticker=tk, leg="CSP", open_date=date.today(),
                    strike=strike, expiry=date.today(), contracts=int(qty),
                    credit_open=credit))
                st.success(f"Added {tk}.")
                st.rerun()
