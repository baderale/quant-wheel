"""Backtest metrics tests (pure functions, no simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantwheel.core.backtest import metrics as M


def _curve(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx)


def test_rising_curve_has_positive_sharpe_and_cagr():
    eq = _curve(list(np.linspace(1.0, 1.5, 260)))
    m = M.equity_metrics(eq)
    assert m["sharpe"] > 0
    assert m["cagr"] > 0
    assert m["total_return"] > 0.4


def test_drawdown_is_detected():
    eq = _curve([1.0, 1.2, 0.9, 1.1])  # peak 1.2 -> trough 0.9 = -25%
    m = M.equity_metrics(eq)
    assert m["max_drawdown"] <= -0.24


def test_aggregate_equity_averages_curves():
    a = _curve([1.0, 1.0, 1.0])
    b = _curve([1.0, 2.0, 3.0])
    agg = M.aggregate_equity([a, b])
    assert agg.iloc[0] == 1.0
    assert agg.iloc[-1] > 1.0


def test_win_rate():
    assert M.win_rate(10, 7) == 0.7
    assert M.win_rate(0, 0) == 0.0


def test_calibration_bins_predictions():
    pairs = [(0.1, 0), (0.12, 0), (0.9, 1), (0.85, 1)]
    cal = M.calibration(pairs, n_bins=5)
    assert len(cal) >= 2
    # Low-prediction bin should have low realized frequency.
    low = min(cal, key=lambda t: t[0])
    assert low[1] <= 0.5
