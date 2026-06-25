"""Monte-Carlo assignment-risk simulation.

Given the underlying's recent behaviour and a candidate strike, simulate many
price paths to expiry and measure:

* ``p_assign``  — fraction of paths ending below the strike (you get assigned).
* ``p_touch``   — fraction of paths that *touch* the strike before expiry
  (mark-to-market pain even if it recovers by Friday).
* ``e_loss_given_assign`` — average per-share intrinsic loss on assigned paths.

Two engines are provided:

* **GBM** (default): increments drawn from a Normal fitted to recent log returns.
* **Bootstrap**: resample *actual* historical log returns with replacement —
  keeps the real fat tails GBM smooths away. Good robustness check.

The path generation is separated from the summary so the *same* simulation can
feed both the assignment stats and the probability-of-profit (which the pricing
layer needs once it knows the break-even). One simulation, two consumers.

Statistical note: with ``n`` paths the standard error on a probability ``p`` is
``sqrt(p(1-p)/n)``. The default 10,000 paths gives ~0.4pp SE at p≈0.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.models import indicators as ind
from quantwheel.core.types import MonteCarloResult

_TRADING_DAYS = 252.0
_CAL_TO_TRADING = 5.0 / 7.0  # rough calendar-day -> trading-day conversion


def trading_steps(dte_calendar: int) -> int:
    """Convert calendar DTE to an integer number of trading steps (>=1)."""
    return max(1, round(dte_calendar * _CAL_TO_TRADING))


@dataclass(frozen=True)
class Paths:
    """Raw simulation output, reusable by multiple summaries."""

    terminal: np.ndarray   # shape (n_paths,) — price at expiry
    path_min: np.ndarray   # shape (n_paths,) — lowest price along each path
    spot: float
    n_steps: int
    method: str

    def prob_above(self, level: float) -> float:
        """Fraction of paths whose terminal price exceeds ``level`` (e.g. POP)."""
        return float(np.mean(self.terminal > level))


def simulate_paths(
    close: pd.Series,
    dte_calendar: int,
    config: WheelConfig = DEFAULT,
    *,
    vol_annual: Optional[float] = None,
    method: str = "gbm",
) -> Paths:
    """Generate price paths to expiry from recent return behaviour.

    ``vol_annual`` overrides the volatility (pass the chain's implied vol on a
    live scan); ``None`` uses realized vol from history (used by the backtest).
    Drift is always estimated from history (real-world measure — we want true
    assignment odds, not the risk-neutral measure used for pricing).
    """
    rets = ind.log_returns(close).dropna().tail(config.mc_return_window)
    if len(rets) < 5:
        raise ValueError("Not enough return history for a Monte-Carlo simulation.")

    spot = float(close.iloc[-1])
    n_steps = trading_steps(dte_calendar)
    rng = np.random.default_rng(config.mc_seed)
    mu_daily = float(rets.mean())

    if method == "bootstrap":
        increments = rng.choice(
            rets.to_numpy(), size=(config.mc_paths, n_steps), replace=True
        )
    elif method == "gbm":
        if vol_annual is not None:
            sigma_daily = vol_annual / np.sqrt(_TRADING_DAYS)
        else:
            sigma_daily = float(rets.std(ddof=1))
        increments = rng.normal(mu_daily, sigma_daily, size=(config.mc_paths, n_steps))
    else:
        raise ValueError(f"Unknown method {method!r}; use 'gbm' or 'bootstrap'.")

    price_paths = np.exp(np.log(spot) + np.cumsum(increments, axis=1))
    return Paths(
        terminal=price_paths[:, -1],
        path_min=price_paths.min(axis=1),
        spot=spot,
        n_steps=n_steps,
        method=method,
    )


def summarize(paths: Paths, strike: float) -> MonteCarloResult:
    """Reduce raw paths to the assignment-risk summary for a given strike."""
    terminal = paths.terminal
    p_assign = float(np.mean(terminal < strike))
    p_touch = float(np.mean(paths.path_min <= strike))
    assigned = terminal[terminal < strike]
    e_loss = float(np.mean(strike - assigned)) if assigned.size else 0.0
    pct = {q: float(np.percentile(terminal, q)) for q in (5, 25, 50, 75, 95)}
    return MonteCarloResult(
        n_paths=int(terminal.size),
        p_assign=p_assign,
        p_touch=p_touch,
        e_loss_given_assign=e_loss,
        terminal_percentiles=pct,
        terminal_mean=float(np.mean(terminal)),
        method=paths.method,
    )


def simulate(
    close: pd.Series,
    strike: float,
    dte_calendar: int,
    config: WheelConfig = DEFAULT,
    *,
    vol_annual: Optional[float] = None,
    method: str = "gbm",
) -> MonteCarloResult:
    """Convenience: generate paths and summarise them for one strike."""
    paths = simulate_paths(
        close, dte_calendar, config, vol_annual=vol_annual, method=method
    )
    return summarize(paths, strike)
