"""Scan service — the facade the UI calls to run a weekly quant scan.

The UI imports *this*, never the models. As long as ``run_weekly_scan`` keeps
returning a :class:`~quantwheel.core.types.ScanResult`, the entire model layer
underneath can be rewritten without the dashboard noticing.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from quantwheel.core import scoring
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import ScanResult
from quantwheel.universe import DEFAULT_UNIVERSE


def run_weekly_scan(
    universe: Optional[list[str]] = None,
    config: WheelConfig = DEFAULT,
    as_of: Optional[date] = None,
) -> ScanResult:
    """Score every ticker in ``universe`` for this week's wheel trade.

    Parameters
    ----------
    universe:
        Tickers to scan. Defaults to the built-in small-account shortlist.
    config:
        Strategy/model parameters. Defaults to the small-account playbook.
    as_of:
        Pin the scan to a date (for reproducible tests). Defaults to today.

    Returns
    -------
    ScanResult
        Candidates sorted best-first, each fully scored and gated.
    """
    universe = universe or DEFAULT_UNIVERSE
    return scoring.score_universe(universe, config, as_of)
