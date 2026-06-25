"""Scan tab — run the quant scan and show the ranked, gated trade sheet."""

from __future__ import annotations

import streamlit as st

from quantwheel.core.narrative import VERDICT_LEGEND
from quantwheel.services.scan_service import run_weekly_scan
from quantwheel.ui.streamlit_app._format import (
    SCAN_COL_FORMAT,
    data_provenance,
    scan_to_df,
)
from quantwheel.universe import DEFAULT_UNIVERSE


def render(config) -> None:
    st.subheader("Weekly scan")
    universe = st.session_state.get("universe", DEFAULT_UNIVERSE)
    st.caption(f"Universe: {', '.join(universe)}  ·  account ${config.account:,.0f}")

    if st.button("▶ Run scan (live market data)", type="primary"):
        with st.spinner("Fetching bars, chains, earnings & scoring…"):
            st.session_state["scan"] = run_weekly_scan(universe, config)

    scan = st.session_state.get("scan")
    if scan is None:
        st.info("Run a scan to see candidates.")
        return

    st.caption(data_provenance(scan))
    df = scan_to_df(scan)
    st.dataframe(
        df.style.format(SCAN_COL_FORMAT, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    # Verdict legend — explain the codes in the table, surfacing those in view first.
    present = list(dict.fromkeys(c.verdict.value for c in scan.candidates))
    with st.expander("🔑 What do the verdicts mean?", expanded=not scan.enterable):
        for v in present + [k for k in VERDICT_LEGEND if k not in present]:
            st.markdown(f"- **{v}** — {VERDICT_LEGEND.get(v, '—')}")

    enter = scan.enterable
    c1, c2 = st.columns(2)
    c1.metric("Enterable", len(enter))
    c2.metric("Scored", len(scan.candidates))
    if enter:
        top = enter[0]
        st.success(
            f"Top pick: **{top.ticker}** — sell {top.sizing.contracts}× "
            f"${top.quote.strike:.2f} put exp {top.quote.expiry} for "
            f"~${top.pricing.credit:.0f} (EV ${top.pricing.expected_value:.1f}, "
            f"POP {top.pricing.pop:.0%})."
        )
    else:
        st.warning("Nothing passed every gate this week — likely capital or regime. "
                   "See the Candidate tab for the per-name reason.")

    if any(c.quote.is_estimated for c in scan.candidates):
        st.caption("⚠️ Rows marked **Est** use a theoretical quote (after-hours / "
                   "synthetic Yahoo data) — verify the live chain in Robinhood.")
