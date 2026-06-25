"""Live option chains from Yahoo Finance, with a graceful theoretical fallback.

The goal is to price against the **real** chain — actual implied vol and bid/ask
— so the modelled credit/delta match Robinhood. But the free Yahoo feed is
unreliable outside regular trading hours: it returns zero bid/ask, zero open
interest, and a synthetic IV "ladder" (0.5, 0.25, 0.125 ...) that is not real.

So this module does the honest thing:

* **If the market is genuinely quoting** (any OTM put has a real ``bid``), use
  the live chain: pick the strike nearest the target delta from each strike's
  own implied vol, and model the credit at the bid/ask mid.
* **Otherwise** (after-hours / stale / synthetic), fall back to a Black-Scholes
  theoretical quote on *realized* vol, **snapped to a real listed strike**, and
  mark it ``is_estimated=True``.

Either way the quote carries everything downstream needs, and an estimated quote
is clearly labelled so the UI can say "verify in Robinhood".
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.models import indicators as ind
from quantwheel.core.models.pricing import bs_put_greeks, bs_put_price, strike_for_delta
from quantwheel.core.types import OptionQuote

# Plausible IV band — reject Yahoo's near-zero / absurd synthetic IVs.
_MIN_IV = 0.05
_MAX_IV = 5.0


@lru_cache(maxsize=64)
def list_expiries(ticker: str) -> tuple[date, ...]:
    """All listed expiration dates for ``ticker`` (ascending)."""
    opts = yf.Ticker(ticker.upper()).options or ()
    return tuple(datetime.strptime(o, "%Y-%m-%d").date() for o in opts)


def pick_expiry(
    ticker: str, config: WheelConfig = DEFAULT, as_of: Optional[date] = None
) -> Optional[date]:
    """Choose the listed expiry whose DTE best fits the weekly window.

    Prefers expiries inside ``[min_dte, max_dte]``; otherwise the nearest expiry
    at least ``min_dte`` out (never something expiring tomorrow). ``None`` if the
    name has no options.
    """
    as_of = as_of or date.today()
    expiries = list_expiries(ticker)
    if not expiries:
        return None

    target = (config.min_dte + config.max_dte) / 2.0
    in_window = [e for e in expiries if config.min_dte <= (e - as_of).days <= config.max_dte]
    pool = in_window or [e for e in expiries if (e - as_of).days >= config.min_dte]
    if not pool:
        return None
    return min(pool, key=lambda e: abs((e - as_of).days - target))


@lru_cache(maxsize=128)
def _puts_for(ticker: str, expiry_iso: str) -> pd.DataFrame:
    """Cached put chain for one expiry (keyed by iso string so it's hashable)."""
    chain = yf.Ticker(ticker.upper()).option_chain(expiry_iso)
    return chain.puts


def _otm_puts(ticker: str, expiry: date, spot: float) -> pd.DataFrame:
    puts = _puts_for(ticker.upper(), expiry.strftime("%Y-%m-%d"))
    if puts is None or puts.empty:
        raise ValueError(f"No puts for {ticker} {expiry}.")
    otm = puts[puts["strike"] < spot].copy()
    if otm.empty:
        raise ValueError(f"No OTM puts for {ticker} {expiry}.")
    return otm


def _live_quote(
    ticker: str, spot: float, expiry: date, otm: pd.DataFrame, dte: int, config: WheelConfig
) -> Optional[OptionQuote]:
    """Build a quote from genuine market data, or ``None`` if none is available."""
    T = max(dte / 365.0, 1e-6)
    iv = otm["impliedVolatility"].fillna(0.0)
    has_quote = (otm["bid"].fillna(0) > 0) & (otm["ask"].fillna(0) > 0)
    live = otm[has_quote & iv.between(_MIN_IV, _MAX_IV)]
    if live.empty:
        return None  # market not quoting -> caller falls back to theoretical

    live = live.copy()
    live["delta_mag"] = live.apply(
        lambda r: abs(
            bs_put_greeks(spot, float(r["strike"]), T, config.risk_free_rate,
                          float(r["impliedVolatility"])).delta
        ),
        axis=1,
    )
    best = live.iloc[(live["delta_mag"] - config.target_delta).abs().argmin()]
    bid, ask = float(best["bid"]), float(best["ask"])
    return OptionQuote(
        ticker=ticker.upper(), expiry=expiry, dte=dte, strike=float(best["strike"]),
        bid=bid, ask=ask, mid=(bid + ask) / 2.0, iv=float(best["impliedVolatility"]),
        open_interest=int(best.get("openInterest", 0) or 0),
        volume=int(best.get("volume", 0) or 0),
        delta=-float(best["delta_mag"]), is_estimated=False, source="yfinance:live",
    )


def _theoretical_quote(
    ticker: str, spot: float, expiry: date, otm: pd.DataFrame, dte: int,
    close: pd.Series, config: WheelConfig,
) -> OptionQuote:
    """Black-Scholes quote on realized vol, snapped to a real listed strike.

    Used when the live chain is unusable. Real *strikes* are still trustworthy
    even when Yahoo's quotes are not, so we snap the delta-target strike to the
    nearest listed OTM strike and price it ourselves.
    """
    T = max(dte / 365.0, 1e-6)
    r = config.risk_free_rate
    sigma = ind.realized_vol(close, window=config.mc_return_window)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.30  # last-resort default if history is too short

    raw_strike = strike_for_delta(spot, T, r, sigma, config.target_delta)
    strikes = otm["strike"].to_numpy(dtype=float)
    strike = float(strikes[np.abs(strikes - raw_strike).argmin()])

    premium = bs_put_price(spot, strike, T, r, sigma)
    delta = bs_put_greeks(spot, strike, T, r, sigma).delta
    row = otm[otm["strike"] == strike]
    return OptionQuote(
        ticker=ticker.upper(), expiry=expiry, dte=dte, strike=strike,
        bid=0.0, ask=0.0, mid=round(premium, 2), iv=sigma,
        open_interest=int(row["openInterest"].iloc[0]) if not row.empty else 0,
        volume=int(row["volume"].iloc[0]) if not row.empty else 0,
        delta=delta, is_estimated=True, source="bs:realized_vol",
    )


def quote_put(
    ticker: str,
    spot: float,
    close: pd.Series,
    expiry: date,
    config: WheelConfig = DEFAULT,
    as_of: Optional[date] = None,
) -> OptionQuote:
    """Best available put quote for ``expiry``: live if quoting, else theoretical."""
    as_of = as_of or date.today()
    dte = (expiry - as_of).days
    otm = _otm_puts(ticker, expiry, spot)
    live = _live_quote(ticker, spot, expiry, otm, dte, config)
    return live or _theoretical_quote(ticker, spot, expiry, otm, dte, close, config)


def fetch_put_quote(
    ticker: str,
    spot: float,
    close: pd.Series,
    config: WheelConfig = DEFAULT,
    as_of: Optional[date] = None,
) -> Optional[OptionQuote]:
    """Convenience: pick the expiry, then the best quote. ``None`` if no options."""
    expiry = pick_expiry(ticker, config, as_of)
    if expiry is None:
        return None
    return quote_put(ticker, spot, close, expiry, config, as_of)
