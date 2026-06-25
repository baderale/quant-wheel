"""Portfolio allocator tests — pure math, no network."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from quantwheel.core import portfolio
from quantwheel.core.config import DEFAULT
from quantwheel.core.types import ScanResult


def _mk(base, ticker, strike):
    """Clone the fixture candidate with a new ticker/strike (still enterable)."""
    q = replace(base.quote, ticker=ticker, strike=strike)
    return replace(base, ticker=ticker, quote=q)


def _scan(candidates, account=1000.0):
    return ScanResult(as_of=date(2026, 6, 25), account=account,
                      universe=[c.ticker for c in candidates], candidates=candidates)


def test_single_position_matches_legacy_sizing(scored_candidate):
    # account 1000, buffer 300 -> 700 deployable; strike 4.5 -> 450/contract -> 1 fits.
    scan = _scan([scored_candidate])
    positions = portfolio.allocate(scan, DEFAULT)
    assert len(positions) == 1
    assert positions[0].contracts == 1
    assert positions[0].collateral == 450.0


def test_multi_position_spreads_across_names(scored_candidate):
    cands = [_mk(scored_candidate, t, 4.5) for t in ("F", "SOFI", "NIO")]
    scan = _scan(cands, account=25_000.0)
    config = replace(DEFAULT, account=25_000.0, max_positions=3)
    positions = portfolio.allocate(scan, config)

    assert len(positions) == 3
    total = sum(p.collateral for p in positions)
    assert total <= config.deployable_cash * config.max_pos_pct + 1e-6


def test_respects_max_positions_cap(scored_candidate):
    cands = [_mk(scored_candidate, f"T{i}", 4.5) for i in range(5)]
    scan = _scan(cands, account=25_000.0)
    config = replace(DEFAULT, account=25_000.0, max_positions=2)
    positions = portfolio.allocate(scan, config)
    assert len(positions) == 2


def test_skips_names_that_do_not_fit(scored_candidate):
    # A $50 strike needs $5,000 collateral — does not fit a $1,000 account.
    pricey = _mk(scored_candidate, "BIG", 50.0)
    scan = _scan([pricey])
    positions = portfolio.allocate(scan, DEFAULT)
    assert positions == []
