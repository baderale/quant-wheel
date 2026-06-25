"""Markov regime model.

Idea: label every historical bar with one of four market *regimes*, learn how
the market transitions between them (a 4x4 transition matrix), then project that
matrix forward to the option's expiry to estimate the probability the regime
stays *favourable* for selling puts.

This is the layer that kills a technically-perfect setup sitting inside a
downtrend — a picture-perfect oversold bounce in a ``TREND_DOWN`` regime is
still a bad put to sell.

The classifier is intentionally simple and *explainable* (a transparent rule on
SMA/RSI/ATR), not a black box. ``regime_sklearn.py`` offers an optional learned
alternative for comparison; the backtest decides which is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.models import indicators as ind
from quantwheel.core.types import RegimeAnalysis, RegimeState

# Fixed ordering so matrix index i <-> state. Do not reorder casually.
STATES: list[RegimeState] = [
    RegimeState.TREND_DOWN,
    RegimeState.VOLATILE,
    RegimeState.RANGE,
    RegimeState.TREND_UP,
]
_INDEX = {s: i for i, s in enumerate(STATES)}


def classify_series(df: pd.DataFrame, config: WheelConfig = DEFAULT) -> pd.Series:
    """Label each bar with a :class:`RegimeState`.

    Rules (checked in order):

    * **VOLATILE** if ATR is above its trailing median (elevated volatility).
    * else **TREND_UP** if price is above the SMA50 and RSI >= 50.
    * else **TREND_DOWN** if price is below the SMA50 and RSI < 50.
    * else **RANGE** (low-vol, sideways — the wheel's happy place).

    Parameters
    ----------
    df:
        OHLC(V) frame with at least ``high``, ``low``, ``close`` columns.
    """
    close = df["close"]
    sma50 = ind.sma(close, config.sma_period)
    rsi = ind.rsi(close, config.rsi_period)
    atr = ind.atr(df["high"], df["low"], close, config.atr_period)
    atr_med = atr.rolling(config.atr_median_window, min_periods=config.atr_period).median()

    out = pd.Series(index=df.index, dtype="object")
    volatile = atr > atr_med
    up = (close > sma50) & (rsi >= 50)
    down = (close < sma50) & (rsi < 50)

    out[:] = RegimeState.RANGE
    out[down] = RegimeState.TREND_DOWN
    out[up] = RegimeState.TREND_UP
    out[volatile] = RegimeState.VOLATILE  # volatility overrides trend label
    # Bars without enough history for the indicators are undefined.
    valid = sma50.notna() & rsi.notna() & atr_med.notna()
    return out.where(valid)


def fit_transition_matrix(
    states: pd.Series, alpha: float = DEFAULT.laplace_alpha
) -> np.ndarray:
    """Estimate the row-stochastic 4x4 transition matrix with Laplace smoothing.

    ``alpha`` (additive/Laplace smoothing) ensures every transition has non-zero
    probability even if it was never observed in the lookback window, so matrix
    powers stay well-behaved.
    """
    counts = np.full((4, 4), alpha, dtype=float)
    clean = states.dropna().tolist()
    for a, b in zip(clean[:-1], clean[1:]):
        counts[_INDEX[a], _INDEX[b]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def project(matrix: np.ndarray, horizon: int) -> np.ndarray:
    """Return ``matrix ** horizon`` (the n-step transition matrix)."""
    return np.linalg.matrix_power(matrix, max(1, int(horizon)))


def analyze_regime(
    df: pd.DataFrame,
    horizon_bars: int,
    config: WheelConfig = DEFAULT,
) -> RegimeAnalysis:
    """Full regime analysis for one ticker: classify -> fit -> project.

    Parameters
    ----------
    df:
        OHLC frame, most recent bar last.
    horizon_bars:
        How many *trading* bars forward to project (≈ DTE in trading days).
    """
    states = classify_series(df, config).dropna().tail(config.regime_lookback)
    if states.empty:
        raise ValueError("Not enough history to classify any regime bars.")

    matrix = fit_transition_matrix(states, config.laplace_alpha)
    current = states.iloc[-1]
    cur_idx = _INDEX[current]

    projected = project(matrix, horizon_bars)
    horizon_row = projected[cur_idx]
    horizon_dist = {STATES[j]: float(horizon_row[j]) for j in range(4)}

    p_favorable = horizon_dist[RegimeState.RANGE] + horizon_dist[RegimeState.TREND_UP]
    p_ii = float(matrix[cur_idx, cur_idx])
    expected_dwell = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float(config.regime_lookback)
    stability = float(np.mean(np.diag(matrix)))

    return RegimeAnalysis(
        current_state=current,
        transition_matrix=matrix.tolist(),
        horizon_distribution=horizon_dist,
        p_favorable=p_favorable,
        expected_dwell_days=expected_dwell,
        stability=stability,
        horizon_days=horizon_bars,
    )
