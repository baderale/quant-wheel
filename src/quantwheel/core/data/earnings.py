"""Next earnings date from Yahoo Finance.

The single most important small-account rule is *never sell a weekly whose
expiry spans an earnings date*. One earnings gap can erase weeks of premium on a
one-position account. This module surfaces the next earnings date so the gate can
enforce that rule.

Yahoo's earnings endpoints are occasionally flaky or empty; every lookup is
wrapped so a failure degrades to ``None`` (treated conservatively by the gate)
rather than crashing the scan.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=64)
def historical_earnings_dates(ticker: str, limit: int = 24) -> tuple[date, ...]:
    """All known earnings dates (past and future) for the backtest's gate.

    Prefers Yahoo's real earnings table. When that is empty (common on the free
    feed), falls back to **synthesizing approximate quarterly dates** stepping
    ~91 days out from the next known earnings date. This is an approximation —
    real dates drift by days/weeks — but it lets the backtest's earnings gate be
    active and the ablation meaningful, instead of a silent no-op.
    """
    try:
        df = yf.Ticker(ticker.upper()).get_earnings_dates(limit=limit)
        if df is not None and not df.empty:
            return tuple(sorted(pd.to_datetime(df.index).tz_localize(None).normalize().date))
    except Exception:
        pass

    # Fallback: quarterly synthesis from the next known earnings date.
    anchor = next_earnings_date(ticker)
    if anchor is None:
        return ()
    synth = [anchor + pd.Timedelta(days=91 * k) for k in range(-12, 5)]
    return tuple(sorted(d.date() if hasattr(d, "date") else d for d in synth))


@lru_cache(maxsize=128)
def next_earnings_date(ticker: str, as_of: Optional[date] = None) -> Optional[date]:
    """Return the next earnings date on/after ``as_of`` (today), or ``None``.

    Tries ``get_earnings_dates`` first (richest source), then the ``calendar``
    fallback. Returns ``None`` if neither yields a future date.
    """
    as_of = as_of or date.today()
    tk = yf.Ticker(ticker.upper())

    # Primary source: a table of recent + upcoming earnings timestamps.
    try:
        df = tk.get_earnings_dates(limit=12)
        if df is not None and not df.empty:
            dates = pd.to_datetime(df.index).tz_localize(None).normalize().date
            future = sorted(d for d in dates if d >= as_of)
            if future:
                return future[0]
    except Exception:
        pass

    # Fallback: the calendar dict.
    try:
        cal = tk.calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            first = ed[0] if isinstance(ed, (list, tuple)) else ed
            d = pd.to_datetime(first).date()
            if d >= as_of:
                return d
    except Exception:
        pass

    return None
