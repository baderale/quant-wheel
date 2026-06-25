"""Technical indicators — vectorised pandas, no loops.

These are the classic momentum/trend/volatility measures the wheel screen uses.
Each function takes pandas Series and returns a pandas Series aligned to the
input index, so they compose cleanly and are trivial to unit-test against known
values.

Definitions follow the standard conventions (Wilder's smoothing for RSI/ATR),
matching what a broker platform shows, so the numbers are comparable to what you
see in Robinhood / TradingView.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over ``period`` bars."""
    return series.rolling(window=period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (0-100).

    RSI > 70 is conventionally overbought (bad time to sell a put — little
    premium, downside ahead); RSI < 30 is oversold (often a good wheel entry).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing == exponential MA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When there are no losses RSI is 100 by definition.
    return out.fillna(100.0).where(close.notna())


def bollinger(
    close: pd.Series, period: int = 20, mult: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands and %B.

    Returns a DataFrame with columns ``mid``, ``upper``, ``lower``, ``pct_b``.
    ``pct_b`` is where price sits in the band: 0 = lower band, 1 = upper band.
    Low %B (near/below 0) is the canonical "buy the dip / sell a put" zone.
    """
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower).replace(0.0, np.nan)
    pct_b = (close - lower) / width
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b}
    )


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range (Wilder) — an absolute volatility measure in price units."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns, ``ln(P_t / P_{t-1})`` — the basis for drift/vol estimates."""
    return np.log(close / close.shift(1))


def realized_vol(close: pd.Series, window: int = 30, annualize: bool = True) -> float:
    """Realized volatility from the trailing ``window`` of log returns.

    Used as the Black-Scholes / Monte-Carlo vol input **only when a live option
    chain IV is unavailable** (e.g. inside the historical backtest). The live
    scan prefers the chain's implied vol.
    """
    rets = log_returns(close).dropna().tail(window)
    if len(rets) < 2:
        return float("nan")
    sigma = float(rets.std(ddof=1))
    return sigma * np.sqrt(252.0) if annualize else sigma


def gap_pct(open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Overnight gap as a fraction: (open - prev_close) / prev_close."""
    return (open_ - prev_close) / prev_close
