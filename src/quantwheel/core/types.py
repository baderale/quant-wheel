"""Shared data contracts for the entire ``quantwheel`` package.

This module is the *lingua franca* of the project. Every layer speaks these
types and nothing else:

    ui/  ->  services/  ->  core/ + agents/

The UI never reaches into ``core`` internals; it calls a function in
``services`` and receives one of the dataclasses defined here. That is what lets
us swap the UI (Streamlit -> Textual TUI -> web) without touching the math.

Design rules for this file:

* **Standard library only.** No numpy/pandas/langchain imports. Keeping the
  contract dependency-free means *any* layer (including a future thin client)
  can import it instantly and cheaply.
* **Plain data, no behaviour.** These are records, not engines. The logic that
  *produces* them lives in ``core`` and ``agents``.
* **Everything is typed and documented**, because this doubles as the
  human-readable spec of what the system computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RegimeState(str, Enum):
    """The four market regimes the Markov model classifies each bar into.

    Inheriting from ``str`` makes the values JSON-serialisable and printable
    (``RegimeState.RANGE == "RANGE"`` is ``True``), which is convenient when
    these flow through LangGraph state or into the dashboard.
    """

    TREND_DOWN = "TREND_DOWN"  # sustained downtrend — worst regime to sell puts into
    VOLATILE = "VOLATILE"      # high ATR / choppy — fat premium but whipsaw risk
    RANGE = "RANGE"            # mean-reverting sideways — the wheel's happy place
    TREND_UP = "TREND_UP"      # steady uptrend — puts rarely assigned


class Verdict(str, Enum):
    """Final machine verdict for a single candidate after all gates.

    The ``SKIP_*`` variants record *why* a name was rejected so the dashboard
    can explain it instead of silently dropping the ticker.
    """

    ENTER_STRONG = "ENTER_STRONG"
    ENTER = "ENTER"
    WATCH = "WATCH"
    SKIP = "SKIP"                  # score too low
    SKIP_REGIME = "SKIP_REGIME"    # forward regime distribution unfavourable
    SKIP_EARNINGS = "SKIP_EARNINGS"  # expiry would span an earnings date
    SKIP_LIQUIDITY = "SKIP_LIQUIDITY"  # spread/OI fails the liquidity gate
    SKIP_CAPITAL = "SKIP_CAPITAL"  # cannot fit even one contract in the account


class Stance(str, Enum):
    """A council persona's directional opinion on a candidate."""

    STRONG_FOR = "STRONG_FOR"
    FOR = "FOR"
    NEUTRAL = "NEUTRAL"
    AGAINST = "AGAINST"
    STRONG_AGAINST = "STRONG_AGAINST"

    @property
    def score(self) -> float:
        """Map a stance onto ``[-1, +1]`` for numeric aggregation."""
        return {
            Stance.STRONG_FOR: 1.0,
            Stance.FOR: 0.5,
            Stance.NEUTRAL: 0.0,
            Stance.AGAINST: -0.5,
            Stance.STRONG_AGAINST: -1.0,
        }[self]


# ---------------------------------------------------------------------------
# Market data / model outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionQuote:
    """A single put option quote pulled from the live chain (yfinance).

    This is the *real* market data that fixes gap #1 in the old Node tool: we
    price against the chain's actual implied volatility and bid/ask midpoint
    instead of a realized-vol proxy.
    """

    ticker: str
    expiry: date
    dte: int                 # calendar days to expiry
    strike: float
    bid: float
    ask: float
    mid: float               # (bid + ask) / 2 — the price we model the credit at
    iv: float                # implied volatility (annualised, e.g. 0.32)
    open_interest: int
    volume: int
    delta: Optional[float] = None  # chain delta if provided, else from our BS model
    # True when the live chain was unusable (after-hours / synthetic IV / no
    # quotes) and we fell back to a Black-Scholes theoretical price on realized
    # vol, snapped to the real listed strike. ALWAYS re-verify in the broker.
    is_estimated: bool = False
    source: str = "yfinance"

    @property
    def spread_pct(self) -> float:
        """Bid/ask spread as a fraction of mid — the liquidity gate input.

        Returns ``nan`` when there is no live two-sided market (bid/ask absent),
        so the liquidity gate can distinguish "wide spread" from "no data".
        """
        if self.bid > 0 and self.ask > 0 and self.mid > 0:
            return (self.ask - self.bid) / self.mid
        return float("nan")


@dataclass(frozen=True)
class RegimeAnalysis:
    """Output of the Markov regime model for one ticker."""

    current_state: RegimeState
    transition_matrix: list[list[float]]      # 4x4, rows sum to 1
    horizon_distribution: dict[RegimeState, float]  # P(state) after `horizon` bars
    p_favorable: float                        # P(RANGE) + P(TREND_UP) at horizon
    expected_dwell_days: float                # avg consecutive days in current state
    stability: float                          # diagonal mass of the matrix (0..1)
    horizon_days: int


@dataclass(frozen=True)
class MonteCarloResult:
    """Output of the GBM (and bootstrap) Monte-Carlo strike-risk simulation."""

    n_paths: int
    p_assign: float                   # P(terminal < strike) — forced to buy shares
    p_touch: float                    # P(path touches strike before expiry)
    e_loss_given_assign: float        # mean per-share loss on assigned paths ($)
    terminal_percentiles: dict[int, float]  # {5: .., 25: .., 50: .., 75: .., 95: ..}
    terminal_mean: float
    method: str = "gbm"               # "gbm" or "bootstrap"


@dataclass(frozen=True)
class Greeks:
    """Black-Scholes Greeks for the modelled put."""

    delta: float      # negative for a short put's underlying sensitivity
    theta: float      # $/day time decay (positive = decay works for the seller)
    vega: float       # $ per 1 vol point
    gamma: float


@dataclass(frozen=True)
class Pricing:
    """Theoretical economics of selling one cash-secured put contract."""

    premium: float        # per share
    credit: float         # per contract (premium * 100)
    breakeven: float      # strike - premium
    max_profit: float     # = credit
    max_loss: float       # (strike - premium) * 100  (stock -> 0)
    required_cash: float  # strike * 100 — the cash-secured collateral
    pop: float            # probability of profit (terminal > breakeven)
    expected_value: float  # corrected EV over the wheel cycle, per contract ($)
    annualized_ror: float  # premium/strike annualised over DTE
    kelly_fraction: float  # optimal capital fraction (capped/clamped elsewhere)
    greeks: Greeks


@dataclass(frozen=True)
class Sizing:
    """How many contracts fit a small one-position account, and why."""

    account: float
    max_pos_pct: float
    contracts: int          # 0 means "does not fit" — see `reason`
    collateral_used: float
    buffer_remaining: float  # cash left idle after the position
    fits: bool
    reason: str             # human-readable explanation (esp. when contracts == 0)


@dataclass(frozen=True)
class GateResult:
    """Result of the hard pre-trade gates (earnings / liquidity / capital)."""

    passed: bool
    earnings_ok: bool
    liquidity_ok: bool
    capital_ok: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredCandidate:
    """Everything the quant core knows about one ticker for this week.

    This is the unit the council personas reason over and the dashboard's
    Candidate Detail tab renders.
    """

    ticker: str
    spot: float
    as_of: date
    indicators: dict[str, float]       # rsi, bb_pct, sma50_dist, atr, ...
    regime: RegimeAnalysis
    monte_carlo: MonteCarloResult
    quote: OptionQuote
    pricing: Pricing
    sizing: Sizing
    gates: GateResult
    score: float                       # composite 0..6 style score
    verdict: Verdict
    next_earnings: Optional[date] = None


@dataclass(frozen=True)
class ScanResult:
    """Output of a full weekly scan across the universe."""

    as_of: date
    account: float
    universe: list[str]
    candidates: list[ScoredCandidate]  # sorted best-first

    @property
    def enterable(self) -> list[ScoredCandidate]:
        """Candidates that passed every gate and earned an ENTER verdict."""
        return [
            c for c in self.candidates
            if c.gates.passed and c.verdict in (Verdict.ENTER, Verdict.ENTER_STRONG)
        ]


# ---------------------------------------------------------------------------
# Council (LangChain personas)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouncilVote:
    """One persona's structured opinion on one candidate."""

    persona: str          # e.g. "Warren Buffett"
    ticker: str
    stance: Stance
    conviction: float     # 0..1 — how strongly the persona holds the stance
    rationale: str        # 1-3 sentence in-character explanation
    key_points: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CouncilVerdict:
    """Aggregated council output: a ranked suggestion list with the votes."""

    ticker: str
    votes: list[CouncilVote]
    aggregate_score: float     # weighted blend of stance*conviction in [-1, 1]
    consensus: str             # short synthesis written by the aggregator
    quant_score: float         # the underlying machine score, for transparency


# ---------------------------------------------------------------------------
# Trade card + tracker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeCard:
    """The human-facing recommendation the user approves before trading.

    Deliberately denormalised and string-friendly: this is what gets rendered
    in the Approve tab and written to the vault. It is a *suggestion only* —
    no code ever places the order.
    """

    ticker: str
    as_of: date
    action: str            # e.g. "Sell cash-secured put"
    strike: float
    expiry: date
    dte: int
    target_credit: float   # per contract
    delta: float
    pop: float
    expected_value: float
    breakeven: float
    contracts: int
    collateral: float
    exit_plan: str         # "Buy to close at ~50% max profit (~$X)"
    assignment_plan: str   # what to do if assigned (covered-call leg)
    council_consensus: str
    rationale: str


@dataclass
class TradeRecord:
    """A row in the built-in trade tracker (mutable — it evolves over a cycle).

    Mirrors the vault ``_Trade Tracker`` note so the dashboard and Obsidian
    stay in sync. ``phase`` tracks where in the wheel cycle we are.
    """

    id: str
    ticker: str
    leg: str                      # "CSP" or "COVERED_CALL"
    open_date: date
    strike: float
    expiry: date
    contracts: int
    credit_open: float            # credit collected when opened
    status: str = "OPEN"          # OPEN | CLOSED | ASSIGNED | EXPIRED | CALLED_AWAY
    phase: str = "PUT"            # PUT | SHARES | CALL  (wheel cycle position)
    close_date: Optional[date] = None
    debit_close: float = 0.0      # cost to buy-to-close (0 if expired)
    realized_pnl: float = 0.0
    cost_basis: Optional[float] = None  # if assigned: strike - cumulative premium
    notes: str = ""


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Output of the walk-forward wheel-cycle backtest."""

    tickers: list[str]
    start: date
    end: date
    equity_curve: list[tuple[str, float]]   # [(iso_date, equity), ...]
    metrics: dict[str, float]               # sharpe, max_drawdown, cagr, win_rate, ...
    calibration: list[tuple[float, float]]  # [(predicted_p_assign, realized_freq), ...]
    ablation: dict[str, dict[str, float]]   # rule_name -> metrics with that rule off
    n_trades: int
