"""Central configuration — every tunable number lives here, once.

The old Node tool scattered magic numbers across files (BB period, RSI window,
delta targets, the $25k account assumption). That made it hard to audit and
impossible to tune consistently. Here they live in one typed place, and the
defaults are taken directly from the vault playbook
(``_Playbook (Small Account Weekly Wheel)``):

* one position at a time, $1,000 account, ~$300 cash buffer
* 7-10 DTE weeklies, 20-25 delta, close at ~50% max profit
* never sell across earnings; require liquid chains

Override any of these by constructing a ``WheelConfig`` and passing it down
through the services layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WheelConfig:
    """Strategy + model parameters. Frozen so it can be safely shared."""

    # --- Account / sizing (the small-account reality) ---------------------
    account: float = 1_000.0
    cash_buffer: float = 300.0       # keep idle to roll / take assignment
    max_pos_pct: float = 1.0         # one position MAY use ~all non-buffer cash
    max_positions: int = 1           # the hard one-position constraint

    # --- Option selection (weekly wheel) ---------------------------------
    target_delta: float = 0.225      # midpoint of the 20-25 delta rule
    delta_tolerance: float = 0.075   # accept 0.15..0.30
    min_dte: int = 5
    max_dte: int = 12                # ~7-10 DTE weeklies, with slack
    profit_take_pct: float = 0.50    # buy-to-close at 50% of max profit

    # --- Gates -----------------------------------------------------------
    earnings_buffer_days: int = 0    # expiry must not be on/after next earnings
    min_open_interest: int = 1_000
    max_spread_pct: float = 0.10     # bid/ask spread <= 10% of mid
    max_assignment_prob: float = 0.40  # downgrade verdict above this MC p_assign

    # --- Technical indicators --------------------------------------------
    bb_period: int = 20
    bb_mult: float = 2.0
    sma_period: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    atr_median_window: int = 60

    # --- Markov regime ----------------------------------------------------
    regime_lookback: int = 252       # bars used to fit the transition matrix
    regime_horizon_days: int = 7     # project forward to ~expiry
    laplace_alpha: float = 1.0       # additive smoothing for unseen transitions

    # --- Monte Carlo ------------------------------------------------------
    mc_paths: int = 10_000           # higher than v1's 5k for tighter SE
    mc_return_window: int = 60       # bars for drift/vol estimate
    mc_seed: int | None = 42         # reproducibility for the repo/notebooks

    # --- Pricing ----------------------------------------------------------
    risk_free_rate: float = 0.04     # annualised; used by Black-Scholes

    # --- Scoring verdict thresholds --------------------------------------
    score_enter_strong: float = 4.5
    score_enter: float = 3.0
    score_watch: float = 1.5
    regime_gate_min: float = 0.50    # p_favorable below this -> SKIP_REGIME

    @property
    def deployable_cash(self) -> float:
        """Cash actually available for collateral after the buffer."""
        return max(0.0, self.account - self.cash_buffer)


# A module-level default so callers can just `from ...config import DEFAULT`.
DEFAULT = WheelConfig()
