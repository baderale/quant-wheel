"""UI-side formatting helpers (presentation only).

These turn the backend's typed objects into display-friendly DataFrames. This is
*presentation* logic, so it lives in the UI layer — the backend never formats for
a screen.
"""

from __future__ import annotations

import pandas as pd

from quantwheel.core.types import ScanResult


def scan_to_df(scan: ScanResult) -> pd.DataFrame:
    """One row per scored candidate, ordered as the scan ranked them."""
    rows = []
    for c in scan.candidates:
        q, p = c.quote, c.pricing
        rows.append({
            "Ticker": c.ticker,
            "Verdict": c.verdict.value,
            "Score": round(c.score, 2),
            "Spot": round(c.spot, 2),
            "Strike": round(q.strike, 2),
            "DTE": q.dte,
            "Credit$": round(p.credit, 0),
            "POP": p.pop,
            "EV$": round(p.expected_value, 1),
            "P(assign)": c.monte_carlo.p_assign,
            "Delta": round(q.delta or 0.0, 3),
            "IV": q.iv,
            "Regime": c.regime.current_state.value,
            "pFav": c.regime.p_favorable,
            "Fits": c.sizing.contracts,
            "Est": "yes" if q.is_estimated else "",
        })
    return pd.DataFrame(rows)


SCAN_COL_FORMAT = {
    "POP": "{:.0%}", "P(assign)": "{:.0%}", "IV": "{:.0%}", "pFav": "{:.0%}",
    "Spot": "${:.2f}", "Strike": "${:.2f}", "Credit$": "${:.0f}", "EV$": "${:.1f}",
}


def data_provenance(scan: ScanResult) -> str:
    """A one-line proof-of-freshness banner: source, timestamp, and any estimates."""
    when = f" · fetched {scan.fetched_at:%H:%M:%S}" if scan.fetched_at else ""
    n_est = sum(1 for c in scan.candidates if c.quote.is_estimated)
    note = "" if n_est == 0 else f" · {n_est} quote(s) estimated (after-hours)"
    return (f"📡 Stock & option data: **live from Yahoo Finance** "
            f"(as of {scan.as_of}{when}){note}.")


def quote_badge(is_estimated: bool) -> str:
    """Per-trade badge distinguishing a real live quote from a theoretical estimate."""
    if is_estimated:
        return ("⚠️ **Estimated quote** (after-hours / synthetic Yahoo data) — "
                "verify the live option chain in your broker before trading.")
    return "✅ **Live option-chain quote** from Yahoo Finance."
