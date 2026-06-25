"""Backtest service — facade for the walk-forward wheel-cycle backtest.

Implemented in Phase 2.
"""

from __future__ import annotations

from datetime import date

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import BacktestResult


def run_backtest(
    tickers: list[str],
    start: date,
    end: date,
    config: WheelConfig = DEFAULT,
) -> BacktestResult:
    """Simulate the full wheel cycle over history and report performance.

    Returns a :class:`BacktestResult` with the equity curve, headline metrics,
    a calibration series (predicted vs realized assignment) and a rule-ablation
    table that quantifies how much each rule (earnings gate, regime gate, 50%
    take) actually contributes.
    """
    from quantwheel.core.backtest.engine import run_backtest as _run

    return _run(tickers, start, end, config)
