"""Council tab — the five personas deliberate and the aggregate ranks names."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantwheel.agents.llm import llm_available
from quantwheel.services.council_service import run_council


def render() -> None:
    st.subheader("Investment council")
    st.caption("Statistician · Buffett · Burry · Dalio · Simons — same evidence, "
               "five judgements, one aggregate.")

    scan = st.session_state.get("scan")
    if scan is None:
        st.info("Run a scan first (Scan tab).")
        return

    ready, reason = llm_available()
    (st.success if ready else st.warning)(f"LLM: {reason}")

    candidates = scan.enterable or scan.candidates[:3]
    st.caption(f"Will deliberate on: {', '.join(c.ticker for c in candidates) or '—'}")

    if st.button("🏛 Convene council", type="primary", disabled=not ready):
        with st.spinner("Five investors deliberating…"):
            st.session_state["council"] = run_council(candidates)

    verdicts = st.session_state.get("council")
    if not verdicts:
        return

    for v in verdicts:
        lean = "🟢" if v.aggregate_score > 0.15 else "🟡" if v.aggregate_score > -0.15 else "🔴"
        with st.expander(f"{lean} {v.ticker} — net {v.aggregate_score:+.2f} "
                         f"(quant score {v.quant_score:.1f})", expanded=True):
            st.write(v.consensus)
            st.dataframe(pd.DataFrame([{
                "Persona": vote.persona, "Stance": vote.stance.value,
                "Conviction": f"{vote.conviction:.0%}", "Rationale": vote.rationale,
            } for vote in v.votes]), hide_index=True, use_container_width=True)
