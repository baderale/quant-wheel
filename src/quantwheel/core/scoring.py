"""Scoring — combine every model layer into one verdict per ticker.

Pipeline for a single name::

    bars -> indicators -> regime -> chain quote -> Monte Carlo -> pricing
         -> earnings/liquidity/capital gates -> composite score -> verdict

The verdict is deliberately *honest*: hard gates (capital, earnings) and a
non-positive expected value override a pretty technical score. A name can look
great and still be ``SKIP_CAPITAL`` because it simply doesn't fit $1,000.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np

from quantwheel.core import sizing as sizing_mod
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.data import bars as bars_mod
from quantwheel.core.data import chains, earnings
from quantwheel.core.models import indicators as ind
from quantwheel.core.models import montecarlo, pricing, regime
from quantwheel.core.types import (
    GateResult,
    ScanResult,
    ScoredCandidate,
    Verdict,
)


def _indicators(df) -> dict[str, float]:
    """Snapshot of the latest technical readings (for display + scoring)."""
    close = df["close"]
    sma50 = ind.sma(close, DEFAULT.sma_period)
    bb = ind.bollinger(close, DEFAULT.bb_period, DEFAULT.bb_mult)
    gaps = ind.gap_pct(df["open"], close.shift(1)).tail(21).abs()
    return {
        "rsi": float(ind.rsi(close).iloc[-1]),
        "bb_pct": float(bb["pct_b"].iloc[-1]),
        "sma50_dist": float(close.iloc[-1] / sma50.iloc[-1] - 1.0),
        "atr": float(ind.atr(df["high"], df["low"], close).iloc[-1]),
        "realized_vol": float(ind.realized_vol(close)),
        "max_gap_21d": float(gaps.max()) if len(gaps) else 0.0,
    }


def _technical_score(feat: dict[str, float]) -> float:
    """Map indicator readings to a 0-6 technical score (4 binary checks, scaled).

    Wheel-entry logic: we want a name that is *not* overbought, sitting in the
    lower half of its range (good entry), not in freefall below trend, and with
    no recent shock gap.
    """
    checks = (
        feat["rsi"] <= 60,                    # not overbought
        feat["bb_pct"] <= 0.5,                # value half of the band
        feat["sma50_dist"] >= -0.05,          # not collapsing below trend
        feat["max_gap_21d"] < 0.10,           # no recent 10%+ shock
    )
    return sum(checks) / 4.0 * 6.0


def _build_gates(
    quote, mc, sizing, next_earnings: Optional[date], config: WheelConfig
) -> GateResult:
    """Apply the hard pre-trade gates and collect human-readable reasons."""
    reasons: list[str] = []

    # Earnings: expiry must not be on/after the next earnings date.
    if next_earnings is None:
        earnings_ok = True
        reasons.append("Earnings date unknown — verify before selling.")
    else:
        earnings_ok = next_earnings > quote.expiry
        if not earnings_ok:
            reasons.append(
                f"Earnings {next_earnings} on/before expiry {quote.expiry} — sit out."
            )

    # Liquidity: Yahoo OI is unreliable (often 0). Treat 0 as 'unknown' rather
    # than failing; only fail on positive-but-thin OI or a measurably wide spread.
    if quote.is_estimated:
        liquidity_ok = True
        reasons.append("Quote estimated (after-hours/synthetic) — verify live chain.")
    else:
        oi_ok = quote.open_interest == 0 or quote.open_interest >= config.min_open_interest
        spread = quote.spread_pct
        spread_ok = not np.isfinite(spread) or spread <= config.max_spread_pct
        liquidity_ok = oi_ok and spread_ok
        if quote.open_interest == 0:
            reasons.append("Open interest unknown — verify in broker.")
        if not oi_ok:
            reasons.append(f"Thin open interest ({quote.open_interest}).")
        if not spread_ok:
            reasons.append(f"Wide spread ({spread:.0%}).")

    capital_ok = sizing.fits
    if not capital_ok:
        reasons.append(sizing.reason)

    passed = earnings_ok and liquidity_ok and capital_ok
    return GateResult(passed, earnings_ok, liquidity_ok, capital_ok, reasons)


def _decide_verdict(score, gates, regime_an, mc, pr, config) -> Verdict:
    """Resolve the final verdict; hard failures and bad EV override the score."""
    if not gates.capital_ok:
        return Verdict.SKIP_CAPITAL
    if not gates.earnings_ok:
        return Verdict.SKIP_EARNINGS
    if regime_an.p_favorable < config.regime_gate_min:
        return Verdict.SKIP_REGIME
    if not gates.liquidity_ok:
        return Verdict.SKIP_LIQUIDITY
    if pr.expected_value <= 0:
        return Verdict.SKIP

    # Aggressive strike (high assignment risk) caps the verdict at WATCH.
    ceiling = Verdict.WATCH if mc.p_assign > config.max_assignment_prob else None
    if score >= config.score_enter_strong:
        decided = Verdict.ENTER_STRONG
    elif score >= config.score_enter:
        decided = Verdict.ENTER
    elif score >= config.score_watch:
        decided = Verdict.WATCH
    else:
        decided = Verdict.SKIP
    if ceiling and decided in (Verdict.ENTER, Verdict.ENTER_STRONG):
        return ceiling
    return decided


def score_candidate(
    ticker: str, config: WheelConfig = DEFAULT, as_of: Optional[date] = None
) -> ScoredCandidate:
    """Run the full pipeline for one ticker and return a scored candidate."""
    as_of = as_of or date.today()
    df = bars_mod.fetch_bars(ticker, period="2y")
    close = df["close"]
    spot = float(close.iloc[-1])

    expiry = chains.pick_expiry(ticker, config, as_of)
    if expiry is None:
        raise ValueError(f"{ticker} has no listed options.")
    quote = chains.quote_put(ticker, spot, close, expiry, config, as_of)

    horizon = montecarlo.trading_steps(quote.dte)
    regime_an = regime.analyze_regime(df, horizon_bars=horizon, config=config)

    paths = montecarlo.simulate_paths(close, quote.dte, config, vol_annual=quote.iv)
    mc = montecarlo.summarize(paths, quote.strike)
    pr = pricing.build_pricing(spot, quote.strike, quote.dte, quote.iv, quote.mid,
                               paths, mc, config)

    next_earn = earnings.next_earnings_date(ticker, as_of)
    sizing = sizing_mod.size_position(pr.required_cash, config)
    gates = _build_gates(quote, mc, sizing, next_earn, config)

    feat = _indicators(df)
    score = _technical_score(feat)
    verdict = _decide_verdict(score, gates, regime_an, mc, pr, config)

    return ScoredCandidate(
        ticker=ticker.upper(), spot=spot, as_of=as_of, indicators=feat,
        regime=regime_an, monte_carlo=mc, quote=quote, pricing=pr, sizing=sizing,
        gates=gates, score=score, verdict=verdict, next_earnings=next_earn,
    )


def _rank_key(c: ScoredCandidate) -> tuple:
    """Sort best-first: gated+enterable first, then by EV, then by score."""
    enter_rank = {
        Verdict.ENTER_STRONG: 0, Verdict.ENTER: 1, Verdict.WATCH: 2,
    }.get(c.verdict, 3)
    return (enter_rank, -c.pricing.expected_value, -c.score)


def score_universe(
    tickers: list[str], config: WheelConfig = DEFAULT, as_of: Optional[date] = None
) -> ScanResult:
    """Score every ticker and return a ranked :class:`ScanResult`.

    Individual ticker failures (bad symbol, data outage) are skipped rather than
    aborting the whole scan.
    """
    as_of = as_of or date.today()
    scored: list[ScoredCandidate] = []
    for t in tickers:
        try:
            scored.append(score_candidate(t, config, as_of))
        except Exception as exc:  # noqa: BLE001 — one bad ticker must not kill the scan
            print(f"  ! skipped {t}: {exc}")
    scored.sort(key=_rank_key)
    return ScanResult(as_of=as_of, account=config.account,
                      universe=[t.upper() for t in tickers], candidates=scored)
