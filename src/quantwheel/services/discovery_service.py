"""Discovery service — reverse search for names that fit the user's account.

When the watchlist yields nothing tradeable, screen the market for affordable,
liquid names and score them with the *same* engine the watchlist uses. Returns a
normal :class:`ScanResult`, so the dashboard can render and report on discovered
names identically to a regular scan.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from quantwheel.core import scoring
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.data import screener
from quantwheel.core.types import ScanResult

DEFAULT_MAX_SCAN = 25   # cap the deep scan after the cheap screener prefilter


def discover(
    config: WheelConfig = DEFAULT,
    exclude: tuple[str, ...] = (),
    max_scan: int = DEFAULT_MAX_SCAN,
    as_of: Optional[date] = None,
) -> ScanResult:
    """Screen for affordable names and score them.

    Parameters
    ----------
    config:
        Drives both the affordability prefilter and the scoring/sizing.
    exclude:
        Tickers to skip (typically the user's existing watchlist).
    max_scan:
        Hard cap on how many screened names get the full (slow) scan.

    Returns
    -------
    ScanResult
        Best-first, identical in shape to a watchlist scan. Empty candidates if
        the screener finds nothing affordable (or is unavailable).
    """
    symbols = screener.find_affordable_optionable(config, limit=max_scan, exclude=exclude)
    if not symbols:
        return ScanResult(as_of=as_of or date.today(), account=config.account,
                          universe=[], candidates=[])
    return scoring.score_universe(symbols, config, as_of)
