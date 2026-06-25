"""Report tab — the plain-language, account-aware recommendation.

Reads the latest scan (and the council, if convened), builds a multi-position
:class:`PortfolioReport` via the report service, and renders it for a beginner or
intermediate investor: a portfolio summary, then each specific trade explained in
plain English with the council's reasoning. Per-trade 'Log to tracker' buttons
reuse the tracker service — nothing here places an order.

When nothing on the watchlist fits, it offers **discovery** (reverse search):
screen the market for affordable, optionable names that do fit the account.
"""

from __future__ import annotations

import streamlit as st

from quantwheel.core.narrative import GLOSSARY
from quantwheel.services import tracker_service
from quantwheel.services.discovery_service import discover
from quantwheel.services.report_service import build_report
from quantwheel.ui.streamlit_app._format import (
    SCAN_COL_FORMAT,
    data_provenance,
    quote_badge,
    scan_to_df,
)


def _glossary() -> None:
    with st.expander("📖 What do these terms mean?"):
        for term, meaning in GLOSSARY.items():
            st.markdown(f"- **{term}** — {meaning}")


def _render_report(report, *, key_prefix: str) -> None:
    """Render a PortfolioReport's summary + per-trade explanations."""
    cols = st.columns(4)
    cols[0].metric("Trades", len(report.items))
    cols[1].metric("Collateral used", f"${report.cash_deployed:,.0f}")
    cols[2].metric("Premium collected", f"${report.total_credit:,.0f}")
    cols[3].metric("Avg. chance to keep it", f"{report.blended_pop:.0%}")
    st.caption(f"Cash kept in reserve: ${report.cash_reserved:,.0f} of "
               f"${report.account:,.0f}.")
    st.divider()

    for i, item in enumerate(report.items):
        card = item.card
        st.markdown(f"### {card.ticker} — {item.summary}")
        st.markdown(quote_badge(item.is_estimated))
        for para in item.explanation:
            st.markdown(para)
        if item.council_explanation:
            st.info(f"**Why the council likes it:** {item.council_explanation}")
        if st.button(f"✅ Log {card.ticker} to tracker",
                     key=f"log_{key_prefix}_{i}_{card.ticker}"):
            rec = tracker_service.add_trade(tracker_service.record_from_card(card))
            st.success(f"Logged {rec.ticker} (id {rec.id}). See the Tracker tab.")
        st.divider()

    st.caption("⚠️ Educational only — no order is placed by software. Verify the "
               "live option chain in your broker before trading.")


def _discovery_section(config, watchlist) -> None:
    """Reverse search: when the watchlist comes up empty, find names that fit."""
    st.divider()
    st.markdown("#### 🔍 Find names that fit your account")
    st.caption(
        f"None of your watchlist cleared the checks for your ${config.account:,.0f} "
        "account. Search the market for affordable, liquid, optionable names — "
        "scored by the same engine."
    )
    if st.button("🔍 Search the market for candidates", type="primary"):
        with st.spinner("Screening the market and scoring candidates…"):
            st.session_state["discovery"] = discover(
                config, exclude=tuple(watchlist), max_scan=25)

    disc = st.session_state.get("discovery")
    if disc is None:
        return
    if not disc.candidates:
        st.info("The screener didn't return any optionable names that fit right "
                "now — try adjusting your cash buffer, or check back during market "
                "hours when more chains are quoting.")
        return

    st.caption(data_provenance(disc))
    disc_report = build_report(disc, None, config)
    st.markdown(disc_report.headline)
    if disc_report.items:
        _render_report(disc_report, key_prefix="disc")
    else:
        st.info("Found affordable names, but none cleared every safety check this "
                "week. The closest matches the engine scored:")
        st.dataframe(
            scan_to_df(disc).head(10).style.format(SCAN_COL_FORMAT, na_rep="—"),
            use_container_width=True, hide_index=True,
        )


def render(config) -> None:
    st.subheader("Your weekly report")
    scan = st.session_state.get("scan")
    if scan is None:
        st.info("Run a scan first (Scan tab) — the report is built from it.")
        return

    report = build_report(scan, st.session_state.get("council"), config)
    st.caption(data_provenance(scan))
    st.markdown(report.headline)

    if report.items:
        _render_report(report, key_prefix="wl")
    else:
        st.warning(report.sit_out_reason)
        _discovery_section(config, scan.universe)

    _glossary()
