"""Monte-Carlo simulation tests."""

from __future__ import annotations

from quantwheel.core.config import DEFAULT
from quantwheel.core.models import montecarlo as mc


def test_probabilities_in_unit_interval(synth_bars):
    close = synth_bars["close"]
    res = mc.simulate(close, strike=float(close.iloc[-1]) * 0.95, dte_calendar=7)
    assert 0.0 <= res.p_assign <= 1.0
    assert 0.0 <= res.p_touch <= 1.0
    assert res.p_touch >= res.p_assign  # touching is necessary for ending below


def test_deeper_otm_lowers_assignment_risk(synth_bars):
    close = synth_bars["close"]
    spot = float(close.iloc[-1])
    near = mc.simulate(close, strike=spot * 0.98, dte_calendar=7)
    far = mc.simulate(close, strike=spot * 0.85, dte_calendar=7)
    assert far.p_assign <= near.p_assign


def test_seed_makes_it_reproducible(synth_bars):
    close = synth_bars["close"]
    a = mc.simulate(close, strike=float(close.iloc[-1]) * 0.95, dte_calendar=7)
    b = mc.simulate(close, strike=float(close.iloc[-1]) * 0.95, dte_calendar=7)
    assert a.p_assign == b.p_assign  # deterministic given config.mc_seed


def test_higher_vol_raises_assignment_risk(synth_bars):
    close = synth_bars["close"]
    strike = float(close.iloc[-1]) * 0.95
    lo = mc.simulate(close, strike, dte_calendar=7, vol_annual=0.15)
    hi = mc.simulate(close, strike, dte_calendar=7, vol_annual=0.80)
    assert hi.p_assign >= lo.p_assign


def test_prob_above_is_monotone(synth_bars):
    paths = mc.simulate_paths(synth_bars["close"], dte_calendar=7)
    assert paths.prob_above(0.0) == 1.0
    assert paths.prob_above(1e12) == 0.0


def test_trading_steps_conversion():
    assert mc.trading_steps(7) >= 1
    assert mc.trading_steps(30) > mc.trading_steps(7)
