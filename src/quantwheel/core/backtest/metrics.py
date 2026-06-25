"""Performance + calibration metrics for the backtest.

Pure functions over an equity curve and the trade log — no simulation here, so
they're trivial to test and reuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252.0


def equity_metrics(equity: pd.Series, rf: float = 0.0) -> dict[str, float]:
    """Headline risk/return metrics from a (normalized) daily equity curve."""
    if equity is None or len(equity) < 3:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "volatility": 0.0}

    rets = equity.pct_change().dropna()
    excess = rets - rf / _TRADING_DAYS
    vol = float(rets.std(ddof=1))
    sharpe = float(excess.mean() / vol * np.sqrt(_TRADING_DAYS)) if vol > 0 else 0.0

    n_days = len(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (_TRADING_DAYS / n_days) - 1.0)
    drawdown = float((equity / equity.cummax() - 1.0).min())

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "total_return": total_return,
        "volatility": float(vol * np.sqrt(_TRADING_DAYS)),
    }


def win_rate(n_trades: int, n_wins: int) -> float:
    """Fraction of closed trades that were profitable."""
    return n_wins / n_trades if n_trades else 0.0


def aggregate_equity(curves: list[pd.Series]) -> pd.Series:
    """Combine per-ticker normalized curves into one equal-weight portfolio curve.

    Aligns on the union of dates, forward-fills gaps, and averages — i.e. an
    equal-weight blend of independent single-name wheels.
    """
    curves = [c for c in curves if c is not None and len(c)]
    if not curves:
        return pd.Series(dtype=float)
    frame = pd.concat(curves, axis=1).sort_index().ffill().dropna(how="all")
    portfolio = frame.mean(axis=1)
    return portfolio / portfolio.iloc[0]


def calibration(pairs: list[tuple[float, int]], n_bins: int = 5) -> list[tuple[float, float]]:
    """Reliability curve: predicted assignment prob vs realized frequency.

    ``pairs`` are ``(predicted_p_assign, realized_assigned 0/1)`` from every put
    that ran to expiry. A well-calibrated model lands near the diagonal.
    """
    if not pairs:
        return []
    df = pd.DataFrame(pairs, columns=["pred", "real"])
    df["bin"] = pd.cut(df["pred"], bins=np.linspace(0, 1, n_bins + 1),
                       include_lowest=True)
    out: list[tuple[float, float]] = []
    for _, g in df.groupby("bin", observed=True):
        if len(g):
            out.append((float(g["pred"].mean()), float(g["real"].mean())))
    return out
