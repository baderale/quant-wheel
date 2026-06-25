"""Black-Scholes pricing/Greeks tests against textbook reference values."""

from __future__ import annotations

import math

from quantwheel.core.models.pricing import (
    bs_put_greeks,
    bs_put_price,
    kelly_fraction,
    strike_for_delta,
)


def test_atm_put_price_matches_reference():
    # Classic reference: S=K=100, T=1, r=5%, sigma=20% -> put = 5.5735.
    price = bs_put_price(100, 100, 1.0, 0.05, 0.20)
    assert math.isclose(price, 5.5735, abs_tol=0.01)


def test_put_call_parity():
    # Recover the call via parity and check against the known call (10.4506).
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    put = bs_put_price(S, K, T, r, sigma)
    call = put + S - K * math.exp(-r * T)
    assert math.isclose(call, 10.4506, abs_tol=0.01)


def test_atm_greeks_reference():
    g = bs_put_greeks(100, 100, 1.0, 0.05, 0.20)
    assert math.isclose(g.delta, -0.3632, abs_tol=0.01)   # N(d1) - 1
    assert math.isclose(g.gamma, 0.01876, abs_tol=0.001)
    assert math.isclose(g.vega, 0.3752, abs_tol=0.01)     # per 1% vol
    assert g.theta < 0                                     # ATM put decays for buyer


def test_deeper_otm_put_is_cheaper():
    near = bs_put_price(100, 95, 0.25, 0.04, 0.30)
    far = bs_put_price(100, 80, 0.25, 0.04, 0.30)
    assert far < near


def test_strike_for_delta_round_trips():
    S, T, r, sigma, target = 100.0, 30 / 365, 0.04, 0.30, 0.225
    K = strike_for_delta(S, T, r, sigma, target)
    assert K < S  # a 22.5-delta put is out-of-the-money
    recovered = abs(bs_put_greeks(S, K, T, r, sigma).delta)
    assert math.isclose(recovered, target, abs_tol=0.01)


def test_kelly_is_clamped_to_unit_interval():
    assert kelly_fraction(0.9, win=50, avg_loss=200) >= 0.0
    assert kelly_fraction(0.99, win=500, avg_loss=1) <= 1.0
    # A coin-flip with symmetric payoff is roughly break-even -> tiny/zero Kelly.
    assert kelly_fraction(0.5, win=100, avg_loss=100) <= 0.05
