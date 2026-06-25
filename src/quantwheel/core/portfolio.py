"""Portfolio allocation — turn a ranked scan into a sized, multi-position book.

A $1,000 one-position account can only ever hold a single cash-secured put, so
the original tool sized one trade against all deployable cash. A larger account
should spread that cash across several names for diversification. This module is
the generalisation: given a scan (and optionally the council's verdicts), pick up
to ``config.max_positions`` enterable candidates and size each within a fair share
of deployable cash, never exceeding the total.

Pure math — no UI, no LLM, no network. For ``config.max_positions == 1`` it is
identical to the original single-position sizing.
"""

from __future__ import annotations

import math
from typing import Optional

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import (
    CouncilVerdict,
    Position,
    ScanResult,
    ScoredCandidate,
)


def _ranked(
    scan: ScanResult, verdicts: list[CouncilVerdict]
) -> list[ScoredCandidate]:
    """Order enterable candidates: council-favorable first, else the scan order."""
    enterable = scan.enterable
    if not verdicts:
        return enterable  # the scan already sorted these best-first
    score_by_ticker = {v.ticker: v.aggregate_score for v in verdicts}
    favorable = sorted(
        (c for c in enterable if score_by_ticker.get(c.ticker, 0.0) > 0),
        key=lambda c: -score_by_ticker.get(c.ticker, 0.0),
    )
    fav_tickers = {c.ticker for c in favorable}
    rest = [c for c in enterable if c.ticker not in fav_tickers]
    return favorable + rest


def allocate(
    scan: ScanResult,
    config: WheelConfig = DEFAULT,
    verdicts: Optional[list[CouncilVerdict]] = None,
) -> list[Position]:
    """Greedily size up to ``config.max_positions`` enterable names.

    Each position is capped at an even share of deployable cash
    (``deployable / max_positions``) so a single pricey name can't swallow the
    whole account; any cash a name doesn't use cascades to the next. Candidates
    that can't fit even one contract in their share are skipped.
    """
    deployable = config.deployable_cash * config.max_pos_pct
    n = max(1, int(config.max_positions))
    budget_each = deployable / n
    remaining = deployable

    positions: list[Position] = []
    for c in _ranked(scan, verdicts or []):
        if len(positions) >= n:
            break
        per_contract = c.quote.strike * 100.0   # cash-secured collateral / contract
        if per_contract <= 0:
            continue
        budget = min(budget_each, remaining)
        contracts = int(math.floor(budget / per_contract))
        if contracts < 1:
            continue
        collateral = contracts * per_contract
        positions.append(
            Position(
                candidate=c,
                contracts=contracts,
                collateral=collateral,
                credit=contracts * c.pricing.credit,
            )
        )
        remaining -= collateral
    return positions
