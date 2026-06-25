"""Walk-forward backtest of the full wheel cycle.

This is the project's honesty layer and the recruiter centerpiece: it simulates
actually *running* the wheel through history and measures whether the rules help.

What it models, week by week, per ticker:

1. **Sell a cash-secured put** ~``target_delta``, ~``min..max_dte`` out, priced
   Black-Scholes on *realized* vol (we have no historical option chains — see the
   limitation note below).
2. Hold, repricing daily. **Take profit** if the put decays to ``profit_take_pct``
   of the entry credit (toggleable).
3. **At expiry**: expire worthless (keep credit) or get **assigned** (buy 100
   shares at the strike; cost basis = strike − premium).
4. **While assigned, sell covered calls** at/above cost basis until the shares are
   **called away**, then return to step 1.

Rules are toggleable so the **ablation** can quantify each one's contribution
(earnings gate, regime gate, profit-take).

Equity is marked daily as ``cash + shares·price − open short-option liability``,
so the curve reflects mark-to-market drawdowns, not just realized cash.

> **Known limitation:** Yahoo provides only *current* option chains, so premiums
> here are Black-Scholes on realized vol, not historical *implied* vol. The live
> tool uses real IV; the backtest approximates it. Real premiums (and thus
> income) are usually a bit higher than modelled, but the *relative* comparison
> the ablation makes is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.models import indicators as ind
from quantwheel.core.models import pricing, regime


@dataclass
class Rules:
    """Toggleable strategy rules (the knobs the ablation flips)."""

    earnings_gate: bool = True
    regime_gate: bool = True
    profit_take: bool = True


@dataclass
class _Leg:
    """An open short option position inside the simulation."""

    kind: str          # "put" or "call"
    strike: float
    entry_idx: int
    expiry_idx: int
    entry_premium: float   # per share
    sigma: float
    predicted_p_assign: float = 0.0


@dataclass
class TickerBacktest:
    """Per-ticker simulation output."""

    ticker: str
    equity: pd.Series                       # daily, normalized to start = 1.0
    realized_assignments: list[tuple[float, int]] = field(default_factory=list)
    n_trades: int = 0
    n_wins: int = 0


def _expiry_index(dates: pd.DatetimeIndex, entry_i: int, config: WheelConfig) -> Optional[int]:
    """Find the bar index closest to the target DTE within the weekly window."""
    entry_date = dates[entry_i]
    target = (config.min_dte + config.max_dte) / 2.0
    best, best_gap = None, 1e9
    for j in range(entry_i + 1, len(dates)):
        dte = (dates[j] - entry_date).days
        if dte < config.min_dte:
            continue
        if dte > config.max_dte:
            break
        gap = abs(dte - target)
        if gap < best_gap:
            best, best_gap = j, gap
    return best


def _spans_earnings(
    dates: pd.DatetimeIndex, i: int, j: int, earnings: tuple[date, ...]
) -> bool:
    """True if any earnings date falls in (entry, expiry]."""
    lo, hi = dates[i].date(), dates[j].date()
    return any(lo < e <= hi for e in earnings)


def backtest_ticker(
    df: pd.DataFrame,
    earnings_dates: tuple[date, ...],
    config: WheelConfig = DEFAULT,
    rules: Rules = Rules(),
    warmup: int = 252,
) -> TickerBacktest:
    """Simulate the wheel on one ticker's daily bars.

    ``df`` must be a date-indexed OHLC frame; the first ``warmup`` bars seed the
    indicators/regime and are not traded.
    """
    dates = df.index
    close = df["close"].to_numpy(dtype=float)
    ticker = str(df.attrs.get("ticker", "?"))
    r = config.risk_free_rate

    # Precompute the regime label for every bar ONCE (classifying indicators per
    # simulated day would be O(n^2)). The gate then just fits a cheap transition
    # matrix on the trailing window.
    states = regime.classify_series(df, config)
    _sidx = {s: k for k, s in enumerate(regime.STATES)}

    def _p_favorable(upto: int, horizon: int) -> float:
        window = states.iloc[: upto + 1].dropna().tail(config.regime_lookback)
        if len(window) < 10:
            return 1.0  # not enough history to judge -> don't block
        P = regime.fit_transition_matrix(window, config.laplace_alpha)
        row = regime.project(P, horizon)[_sidx[window.iloc[-1]]]
        return float(row[_sidx[regime.RegimeState.RANGE]] +
                     row[_sidx[regime.RegimeState.TREND_UP]])

    # Per-ticker notional sized so one contract is always affordable, so the
    # backtest measures *strategy* edge rather than the $1k affordability wall.
    start_capital = float(close[warmup]) * 100.0 * 1.3
    cash = start_capital
    shares = 0
    cost_basis = 0.0
    leg: Optional[_Leg] = None

    equity = np.full(len(close), np.nan)
    bt = TickerBacktest(ticker=ticker, equity=pd.Series(dtype=float))

    i = warmup
    while i < len(close):
        spot = close[i]

        # ---- mark-to-market equity (cash + shares - short-option liability) ----
        liability = 0.0
        if leg is not None:
            T = max((leg.expiry_idx - i) / 365.0 * (7 / 5), 1e-6)  # ~calendar yrs left
            if leg.kind == "put":
                liability = pricing.bs_put_price(spot, leg.strike, T, r, leg.sigma) * 100
            else:
                liability = pricing.bs_call_price(spot, leg.strike, T, r, leg.sigma) * 100
        equity[i] = cash + shares * spot - liability

        # ---------------------------- manage an open leg ----------------------
        if leg is not None:
            cur_val = liability / 100.0
            take = rules.profit_take and cur_val <= config.profit_take_pct * leg.entry_premium
            if take:
                cash -= cur_val * 100.0            # buy to close
                bt.n_trades += 1
                bt.n_wins += 1
                # Record the (correct) calibration outcome: closed early = not
                # assigned. Without this, only losing puts reach expiry and the
                # reliability curve is biased toward "always assigned".
                if leg.kind == "put":
                    bt.realized_assignments.append((leg.predicted_p_assign, 0))
                leg = None
            elif i >= leg.expiry_idx:
                if leg.kind == "put":
                    if spot < leg.strike:          # assigned
                        cash -= leg.strike * 100.0
                        shares = 100
                        cost_basis = leg.strike - leg.entry_premium
                        bt.realized_assignments.append((leg.predicted_p_assign, 1))
                    else:                           # expired worthless
                        bt.n_wins += 1
                        bt.realized_assignments.append((leg.predicted_p_assign, 0))
                    bt.n_trades += 1
                else:  # call
                    if spot >= leg.strike:          # called away
                        cash += leg.strike * 100.0
                        shares = 0
                        bt.n_wins += 1
                    bt.n_trades += 1
                leg = None
            i += 1
            continue

        # ----------------------- open a new leg (flat) ------------------------
        j = _expiry_index(dates, i, config)
        if j is None:
            i += 1
            continue
        dte = (dates[j] - dates[i]).days
        T = max(dte / 365.0, 1e-6)
        hist = pd.Series(close[: i + 1])
        sigma = ind.realized_vol(hist, window=config.mc_return_window)
        if not np.isfinite(sigma) or sigma <= 0:
            i += 1
            continue

        if rules.earnings_gate and _spans_earnings(dates, i, j, earnings_dates):
            i += 1                                   # sit out earnings week
            continue

        if shares == 0:  # PUT phase
            if rules.regime_gate and _p_favorable(i, max(1, dte * 5 // 7)) < config.regime_gate_min:
                i += 1
                continue
            strike = pricing.strike_for_delta(spot, T, r, sigma, config.target_delta)
            strike = round(strike, 1)
            prem = pricing.bs_put_price(spot, strike, T, r, sigma)
            mu = float(ind.log_returns(hist).dropna().tail(config.mc_return_window).mean()) * 252
            p_assign = pricing.prob_terminal_below(spot, strike, T, mu, sigma)
            cash += prem * 100.0
            leg = _Leg("put", strike, i, j, prem, sigma, p_assign)
        else:  # CALL phase — sell covered call at/above cost basis
            strike = pricing.strike_for_delta(spot, T, r, sigma, config.target_delta)
            strike = max(round(strike, 1), round(cost_basis, 1), round(spot, 1))
            prem = pricing.bs_call_price(spot, strike, T, r, sigma)
            cash += prem * 100.0
            leg = _Leg("call", strike, i, j, prem, sigma)
        i += 1

    series = pd.Series(equity, index=dates).dropna()
    bt.equity = series / series.iloc[0] if len(series) else series
    return bt


# ---------------------------------------------------------------------------
# Portfolio orchestration + ablation (this is what the service calls)
# ---------------------------------------------------------------------------

from quantwheel.core.backtest import metrics as _M  # noqa: E402
from quantwheel.core.data import bars as _bars  # noqa: E402
from quantwheel.core.data import earnings as _earn  # noqa: E402
from quantwheel.core.types import BacktestResult  # noqa: E402

_Loaded = list[tuple[pd.DataFrame, tuple]]


def _load(tickers: list[str], start: date, end: date, warmup_days: int = 420) -> _Loaded:
    """Fetch bars + historical earnings once; reused across every ablation run."""
    loaded: _Loaded = []
    lo = pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    hi = pd.Timestamp(end)
    for t in tickers:
        try:
            df = _bars.fetch_bars(t, period="5y")
            df = df[(df.index >= lo) & (df.index <= hi)].copy()
            if len(df) < 300:
                print(f"  ! {t}: too little history in window, skipping")
                continue
            df.attrs["ticker"] = t.upper()
            loaded.append((df, _earn.historical_earnings_dates(t)))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {t}: {exc}")
    return loaded


def _run_variant(loaded: _Loaded, config: WheelConfig, rules: Rules):
    curves, cal_pairs, n_trades, n_wins = [], [], 0, 0
    for df, edates in loaded:
        bt = backtest_ticker(df, edates, config, rules)
        if len(bt.equity):
            curves.append(bt.equity)
        cal_pairs += bt.realized_assignments
        n_trades += bt.n_trades
        n_wins += bt.n_wins
    eq = _M.aggregate_equity(curves)
    m = _M.equity_metrics(eq)
    m["win_rate"] = _M.win_rate(n_trades, n_wins)
    return eq, m, cal_pairs, n_trades


def run_backtest(
    tickers: list[str], start: date, end: date, config: WheelConfig = DEFAULT
) -> BacktestResult:
    """Full backtest: baseline + rule-ablation, returned as a BacktestResult.

    The ablation re-runs the simulation with each rule disabled so the dashboard
    can show how much each one (earnings gate, regime gate, profit-take) actually
    contributes to risk-adjusted return.
    """
    loaded = _load(tickers, start, end)
    if not loaded:
        raise ValueError("No usable tickers loaded for the backtest window.")

    eq, base_metrics, cal_pairs, n_trades = _run_variant(loaded, config, Rules())

    ablation: dict[str, dict[str, float]] = {}
    for name, rules in (
        ("no_earnings_gate", Rules(earnings_gate=False)),
        ("no_regime_gate", Rules(regime_gate=False)),
        ("no_profit_take", Rules(profit_take=False)),
    ):
        _, m, _, _ = _run_variant(loaded, config, rules)
        ablation[name] = m

    return BacktestResult(
        tickers=[t.upper() for t in tickers],
        start=start,
        end=end,
        equity_curve=[(d.strftime("%Y-%m-%d"), float(v)) for d, v in eq.items()],
        metrics=base_metrics,
        calibration=_M.calibration(cal_pairs),
        ablation=ablation,
        n_trades=n_trades,
    )
