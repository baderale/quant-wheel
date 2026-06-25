"""Indicator tests — sanity bounds and known behaviours."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantwheel.core.models import indicators as ind


def test_rsi_of_monotonic_rally_is_high():
    close = pd.Series(np.linspace(10, 20, 60))  # strictly rising
    assert ind.rsi(close).iloc[-1] > 95          # no losses -> RSI -> 100


def test_rsi_stays_in_bounds(synth_bars):
    r = ind.rsi(synth_bars["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_bollinger_pct_b_brackets_price(synth_bars):
    bb = ind.bollinger(synth_bars["close"]).dropna()
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()


def test_realized_vol_positive(synth_bars):
    rv = ind.realized_vol(synth_bars["close"])
    assert rv > 0


def test_atr_is_positive(synth_bars):
    a = ind.atr(synth_bars["high"], synth_bars["low"], synth_bars["close"]).dropna()
    assert (a > 0).all()
