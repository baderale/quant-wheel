"""Black-Scholes pricing, Greeks, and per-trade economics.

Pricing uses the **risk-neutral** Black-Scholes world (that's what option prices
live in), fed the chain's *implied* volatility on a live scan. Assignment
probability, by contrast, comes from the Monte-Carlo layer under the
**real-world** measure — the two measures answer different questions and we keep
them separate on purpose.

The expected-value formula here fixes the bug in the old Node tool, which
silently dropped the collected premium on assigned paths. The honest
single-cycle EV is::

    EV = [ premium  -  E[loss | assigned] * P(assign) ] * 100

i.e. you *always* keep the premium; assignment only subtracts the average
intrinsic shortfall. (The *full* wheel cycle — selling covered calls after
assignment instead of realizing the loss — is modelled in the backtest, not in a
single trade's EV.)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.models.montecarlo import Paths
from quantwheel.core.types import Greeks, MonteCarloResult, Pricing


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """The standard Black-Scholes d1/d2 terms (guards tiny T/sigma)."""
    sig_sqrt_t = max(sigma * np.sqrt(T), 1e-9)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    return d1, d2


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price: ``K e^{-rT} N(-d2) - S N(-d1)``."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price — used for the covered-call leg in the backtest."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def prob_terminal_below(S: float, K: float, T: float, mu: float, sigma: float) -> float:
    """Analytic P(S_T < K) under lognormal dynamics with drift ``mu`` (annualised).

    Used by the backtest's *predicted* assignment probability (for calibration)
    without paying for a Monte-Carlo run on every simulated week.
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if S < K else 0.0
    drift = (mu - 0.5 * sigma**2) * T
    z = (np.log(K / S) - drift) / (sigma * np.sqrt(T))
    return float(norm.cdf(z))


def bs_put_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Greeks:
    """Put Greeks. ``theta`` is per *day*; ``vega`` is per *1% vol point*."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = float(norm.pdf(d1))
    delta = float(norm.cdf(d1) - 1.0)                       # in [-1, 0]
    gamma = pdf_d1 / (S * max(sigma * np.sqrt(T), 1e-9))
    vega = S * pdf_d1 * np.sqrt(T) / 100.0                  # per 1% vol
    theta_year = (
        -(S * pdf_d1 * sigma) / (2.0 * np.sqrt(T))
        + r * K * np.exp(-r * T) * float(norm.cdf(-d2))
    )
    return Greeks(delta=delta, theta=theta_year / 365.0, vega=vega, gamma=gamma)


def strike_for_delta(
    S: float, T: float, r: float, sigma: float, target_delta: float
) -> float:
    """Invert put delta to find the strike with magnitude ``target_delta``.

    Used by the **backtest** (which has no live chain to read strikes from) to
    pick the ~20-25 delta strike. ``target_delta`` is a positive magnitude, e.g.
    ``0.225``. A short put's delta is negative; we solve ``N(-d1) = target``.
    """
    target_delta = min(max(target_delta, 1e-4), 0.999)
    d1 = -float(norm.ppf(target_delta))
    # d1 = (ln(S/K) + (r + 0.5 sigma^2) T) / (sigma sqrt(T))  ->  solve for K
    ln_s_over_k = d1 * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T
    return float(S / np.exp(ln_s_over_k))


def kelly_fraction(pop: float, win: float, avg_loss: float) -> float:
    """Growth-optimal capital fraction for a binary-ish bet, clamped to [0, 1].

    ``win`` and ``avg_loss`` are positive per-contract amounts. This is a rough
    sizing *guide*; the real constraint on this account is one-position-at-a-time
    (see ``sizing.py``), so we never bet the full Kelly.
    """
    if avg_loss <= 0:
        return min(1.0, max(0.0, pop))
    b = win / avg_loss
    f = (b * pop - (1.0 - pop)) / b
    return float(min(1.0, max(0.0, f)))


def build_pricing(
    spot: float,
    strike: float,
    dte_calendar: int,
    sigma_annual: float,
    premium: float,
    paths: Paths,
    mc: MonteCarloResult,
    config: WheelConfig = DEFAULT,
) -> Pricing:
    """Assemble the full :class:`Pricing` for one cash-secured put.

    Parameters
    ----------
    premium:
        The credit per share we model collecting — the chain bid/ask **mid** on
        a live scan, or the Black-Scholes price in the backtest.
    paths:
        The Monte-Carlo paths (reused to compute POP at the break-even).
    mc:
        The summarised Monte-Carlo result (p_assign, e_loss_given_assign).
    """
    r = config.risk_free_rate
    T = max(dte_calendar / 365.0, 1e-6)

    greeks = bs_put_greeks(spot, strike, T, r, sigma_annual)
    breakeven = strike - premium
    credit = premium * 100.0
    required_cash = strike * 100.0

    pop = paths.prob_above(breakeven)
    # Corrected single-cycle EV (premium kept on every path).
    expected_value = (premium - mc.e_loss_given_assign * mc.p_assign) * 100.0
    annualized_ror = (premium / strike) * (365.0 / max(dte_calendar, 1))

    avg_loss = mc.e_loss_given_assign * 100.0
    kelly = kelly_fraction(pop, credit, avg_loss)

    return Pricing(
        premium=premium,
        credit=credit,
        breakeven=breakeven,
        max_profit=credit,
        max_loss=(strike - premium) * 100.0,
        required_cash=required_cash,
        pop=pop,
        expected_value=expected_value,
        annualized_ror=annualized_ror,
        kelly_fraction=kelly,
        greeks=greeks,
    )
