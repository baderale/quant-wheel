"""Shared pytest fixtures.

The math layers are deterministic given their inputs, so we test them against
*synthetic* price data (no network). ``synth_bars`` builds a reproducible
random-walk OHLCV frame shaped exactly like what ``core.data.bars`` returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_bars() -> pd.DataFrame:
    """A deterministic ~400-bar OHLCV frame (seeded), columns lower-cased."""
    rng = np.random.default_rng(0)
    n = 400
    rets = rng.normal(0.0003, 0.015, n)          # mild upward drift, ~24% annual vol
    close = 50.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    noise = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + noise)
    low = np.minimum(open_, close) * (1 - noise)
    volume = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def scored_candidate():
    """A fully-populated ScoredCandidate for council/agent tests (no network)."""
    from datetime import date

    from quantwheel.core.types import (
        GateResult, Greeks, MonteCarloResult, OptionQuote, Pricing,
        RegimeAnalysis, RegimeState, ScoredCandidate, Sizing, Verdict,
    )

    quote = OptionQuote(
        ticker="NIO", expiry=date(2026, 7, 2), dte=7, strike=4.5, bid=0.10,
        ask=0.12, mid=0.11, iv=0.55, open_interest=5000, volume=2000, delta=-0.22,
    )
    greeks = Greeks(delta=-0.22, theta=-0.01, vega=0.004, gamma=0.05)
    pricing = Pricing(
        premium=0.11, credit=11.0, breakeven=4.39, max_profit=11.0, max_loss=439.0,
        required_cash=450.0, pop=0.85, expected_value=6.0, annualized_ror=0.40,
        kelly_fraction=0.3, greeks=greeks,
    )
    regime = RegimeAnalysis(
        current_state=RegimeState.RANGE,
        transition_matrix=[[0.25] * 4] * 4,
        horizon_distribution={s: 0.25 for s in RegimeState},
        p_favorable=0.55, expected_dwell_days=8.0, stability=0.5, horizon_days=5,
    )
    mc = MonteCarloResult(
        n_paths=10000, p_assign=0.17, p_touch=0.25, e_loss_given_assign=0.3,
        terminal_percentiles={5: 3.9, 25: 4.3, 50: 4.6, 75: 4.9, 95: 5.3},
        terminal_mean=4.6,
    )
    return ScoredCandidate(
        ticker="NIO", spot=5.0, as_of=date(2026, 6, 25),
        indicators={"rsi": 45.0, "bb_pct": 0.3, "sma50_dist": -0.02, "atr": 0.2,
                    "realized_vol": 0.55, "max_gap_21d": 0.04},
        regime=regime, monte_carlo=mc, quote=quote, pricing=pricing,
        sizing=Sizing(1000, 1.0, 1, 450.0, 550.0, True, "Fits 1 contract."),
        gates=GateResult(True, True, True, True, []),
        score=4.5, verdict=Verdict.ENTER, next_earnings=date(2026, 8, 1),
    )
