"""Approve tab — the human-in-the-loop trade card. Logs to the tracker on approval.

Nothing here places an order. Approval only records the suggestion you will then
execute manually in Robinhood.
"""

from __future__ import annotations

import streamlit as st

from quantwheel.agents.nodes import make_trade_card
from quantwheel.services import tracker_service


def _pick_candidate(scan, council):
    """Prefer the council's top favorable name; fall back to the scan's top pick."""
    if council:
        favorable = [v for v in council if v.aggregate_score > 0]
        if favorable:
            by_ticker = {c.ticker: c for c in scan.candidates}
            v = favorable[0]
            return by_ticker.get(v.ticker), v
    enter = scan.enterable
    return (enter[0] if enter else None), None


def render(config) -> None:
    st.subheader("Approve & log")
    scan = st.session_state.get("scan")
    if scan is None:
        st.info("Run a scan first (Scan tab).")
        return

    candidate, verdict = _pick_candidate(scan, st.session_state.get("council"))
    if candidate is None:
        st.warning("No enterable candidate to approve this week — sit out.")
        return

    card = make_trade_card(candidate, verdict, config)
    st.markdown(f"### {card.ticker} — {card.action}")
    cols = st.columns(4)
    cols[0].metric("Strike", f"${card.strike:.2f}")
    cols[1].metric("Expiry", str(card.expiry), f"{card.dte} DTE")
    cols[2].metric("Credit", f"${card.target_credit:.0f}", f"×{card.contracts}")
    cols[3].metric("POP / EV", f"{card.pop:.0%}", f"${card.expected_value:.1f}")

    st.markdown(f"**Break-even:** ${card.breakeven:.2f}  ·  "
                f"**Collateral:** ${card.collateral:,.0f}")
    st.markdown(f"**Exit plan:** {card.exit_plan}")
    st.markdown(f"**Assignment plan:** {card.assignment_plan}")
    st.markdown(f"**Why:** {card.rationale}")
    if card.council_consensus and card.council_consensus != "n/a":
        st.info(f"Council: {card.council_consensus}")

    st.divider()
    st.caption("⚠️ Educational only. Verify the live chain in Robinhood, then place the "
               "order yourself. Approving just records it to your tracker.")
    if st.button("✅ Approve & log to tracker", type="primary"):
        rec = tracker_service.add_trade(tracker_service.record_from_card(card))
        st.success(f"Logged trade {rec.id} for {rec.ticker}. See the Tracker tab.")
