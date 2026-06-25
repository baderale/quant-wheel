"""Universe tab — view/edit the watchlist that the scan runs over."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantwheel.universe import DEFAULT_UNIVERSE


def render() -> None:
    st.subheader("Watchlist")
    st.caption(
        "The names the scan considers. Mirrors the vault shortlist — Tier-A "
        "(F, SOFI, T, AAL) are core; Tier-B (NIO, MARA, CLSK) are higher-risk. "
        "At $1,000 only sub-~$7 strikes fit a true cash-secured put."
    )

    current = st.session_state.get("universe", DEFAULT_UNIVERSE)
    df = pd.DataFrame({"ticker": current})
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"ticker": st.column_config.TextColumn("Ticker", required=True)},
    )
    tickers = [str(t).strip().upper() for t in edited["ticker"].dropna() if str(t).strip()]
    st.session_state["universe"] = tickers or DEFAULT_UNIVERSE

    st.success(f"{len(st.session_state['universe'])} tickers ready: "
               f"{', '.join(st.session_state['universe'])}")
    if st.button("Reset to default shortlist"):
        st.session_state["universe"] = list(DEFAULT_UNIVERSE)
        st.rerun()
