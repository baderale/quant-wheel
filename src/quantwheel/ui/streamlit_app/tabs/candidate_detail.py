"""Candidate tab — drill into one name: regime, Monte Carlo, Greeks, gates."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render() -> None:
    st.subheader("Candidate detail")
    scan = st.session_state.get("scan")
    if scan is None:
        st.info("Run a scan first (Scan tab).")
        return

    tickers = [c.ticker for c in scan.candidates]
    pick = st.selectbox("Ticker", tickers)
    c = next(x for x in scan.candidates if x.ticker == pick)
    q, p, r, mc = c.quote, c.pricing, c.regime, c.monte_carlo

    st.markdown(f"### {c.ticker} — `{c.verdict.value}` · score {c.score:.2f}/6")
    cols = st.columns(4)
    cols[0].metric("Spot", f"${c.spot:.2f}")
    cols[1].metric("Strike", f"${q.strike:.2f}", f"{q.dte} DTE")
    cols[2].metric("POP", f"{p.pop:.0%}")
    cols[3].metric("EV / contract", f"${p.expected_value:.1f}")
    cols = st.columns(4)
    cols[0].metric("P(assign)", f"{mc.p_assign:.0%}")
    cols[1].metric("P(touch)", f"{mc.p_touch:.0%}")
    cols[2].metric("Delta", f"{q.delta or 0:.3f}")
    cols[3].metric("Credit", f"${p.credit:.0f}")

    left, right = st.columns(2)
    with left:
        st.caption("Regime distribution at expiry")
        st.bar_chart(pd.Series({s.value: v for s, v in r.horizon_distribution.items()}))
        st.caption(f"Current regime **{r.current_state.value}** · "
                   f"P(favorable) {r.p_favorable:.0%} · stability {r.stability:.2f}")
    with right:
        st.caption("Monte-Carlo terminal price percentiles")
        st.bar_chart(pd.Series({f"p{k}": v for k, v in mc.terminal_percentiles.items()}))
        st.caption(f"{mc.n_paths:,} paths · method {mc.method}")

    st.caption("Greeks")
    g = p.greeks
    st.dataframe(pd.DataFrame([{
        "delta": round(g.delta, 3), "theta/day": round(g.theta, 3),
        "vega/1%": round(g.vega, 3), "gamma": round(g.gamma, 4),
    }]), hide_index=True, use_container_width=True)

    st.caption("Technicals")
    st.json({k: round(v, 3) for k, v in c.indicators.items()})

    st.caption("Gates & sizing")
    st.write("✅ passed all gates" if c.gates.passed else "⛔ blocked")
    for reason in c.gates.reasons:
        st.write(f"- {reason}")
    st.info(c.sizing.reason)
    if q.is_estimated:
        st.warning(f"Quote is a theoretical estimate (source: {q.source}). "
                   f"Verify the live chain before trading.")
