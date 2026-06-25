"""Daily price bars from Yahoo Finance (via :mod:`yfinance`).

Free, no API key. Returns a tidy lower-cased OHLCV frame indexed by date so the
rest of ``core`` never has to know where the data came from — swapping in a paid
feed later means rewriting only this file.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=128)
def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Cached raw download (per process run) to avoid hammering Yahoo."""
    raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    return raw


def fetch_bars(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Return OHLCV bars for ``ticker`` as a lower-cased, date-indexed frame.

    Columns: ``open, high, low, close, volume``. Raises ``ValueError`` if Yahoo
    returns nothing (bad symbol, delisted, or a transient outage).
    """
    raw = _download(ticker.upper(), period, interval)
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {ticker!r}.")

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.dropna(subset=["close"])


def latest_spot(ticker: str) -> float:
    """Most recent close price for ``ticker``."""
    return float(fetch_bars(ticker, period="5d")["close"].iloc[-1])
