"""Report service — assemble the week's plain-language, multi-position recommendation.

This is the facade the dashboard's **Report** tab calls. It runs the portfolio
allocator over a scan (optionally ordered by the council's verdicts), builds a
human-facing :class:`TradeCard` for each sized position, attaches the plain-English
narrative, and rolls everything up into a :class:`PortfolioReport`.

UI-agnostic: it returns dataclasses only, so a different front-end (the Textual
TUI, a web client) could render the same report.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from quantwheel.agents.nodes import make_trade_card
from quantwheel.core import narrative, portfolio
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import (
    CouncilVerdict,
    PortfolioReport,
    ReportItem,
    ScanResult,
)

# Plain-English reason per SKIP verdict, for the "sit out" explanation.
_SKIP_REASON = {
    "SKIP_CAPITAL": "the strike needed more collateral than your account can deploy",
    "SKIP_REGIME": "an unfavorable market regime (the trend/volatility backdrop)",
    "SKIP_EARNINGS": "an earnings report landing on or before the option's expiry",
    "SKIP_LIQUIDITY": "thin or wide option markets (poor liquidity)",
    "SKIP": "a weak setup or non-positive expected value",
}


def _sit_out_reason(scan: ScanResult, config: WheelConfig) -> str:
    """A friendly explanation of why nothing qualified this week."""
    skips = [c.verdict.value for c in scan.candidates if c.verdict.value.startswith("SKIP")]
    if skips:
        top = max(set(skips), key=skips.count)
        why = _SKIP_REASON.get(top, "the safety checks")
        blocker = f" The most common blocker was {why}."
    else:
        blocker = ""
    return (
        f"No trade cleared every safety check for your ${config.account:,.0f} "
        f"account this week.{blocker} Sitting out is a valid move — your cash "
        f"simply waits for a better setup."
    )


def build_report(
    scan: ScanResult,
    council_verdicts: Optional[list[CouncilVerdict]] = None,
    config: WheelConfig = DEFAULT,
) -> PortfolioReport:
    """Turn a scan (+ optional council) into a readable, sized recommendation."""
    deployable = config.deployable_cash
    positions = portfolio.allocate(scan, config, council_verdicts)
    verdict_by_ticker = {v.ticker: v for v in (council_verdicts or [])}

    items: list[ReportItem] = []
    for pos in positions:
        c = pos.candidate
        verdict = verdict_by_ticker.get(c.ticker)
        # Build the card, then reflect the *allocated* size (the allocator may size
        # differently from the per-candidate standalone sizing in the scan).
        card = make_trade_card(c, verdict, config)
        card = replace(card, contracts=pos.contracts, collateral=pos.collateral)
        items.append(
            ReportItem(
                position=pos,
                card=card,
                summary=narrative.summary_line(pos, card),
                explanation=narrative.build_explanation(pos, card, config),
                council_explanation=narrative.explain_council(verdict),
                is_estimated=c.quote.is_estimated,
                source=c.quote.source,
            )
        )

    cash_deployed = sum(i.position.collateral for i in items)
    total_credit = sum(i.position.credit for i in items)
    total_contracts = sum(i.position.contracts for i in items)
    blended_pop = (
        sum(i.card.pop * i.position.contracts for i in items) / total_contracts
        if total_contracts else 0.0
    )

    if items:
        n = len(items)
        headline = (
            f"**{n} trade{'s' if n != 1 else ''}** fit your ${config.account:,.0f} "
            f"account this week — setting aside ${cash_deployed:,.0f} of collateral "
            f"and collecting about ${total_credit:,.0f} in premium."
        )
        sit_out = ""
    else:
        headline = f"**Sit out this week.** Nothing fit your ${config.account:,.0f} account."
        sit_out = _sit_out_reason(scan, config)

    return PortfolioReport(
        as_of=scan.as_of,
        account=config.account,
        deployable=deployable,
        max_positions=config.max_positions,
        items=items,
        cash_deployed=cash_deployed,
        cash_reserved=config.account - cash_deployed,
        total_credit=total_credit,
        blended_pop=blended_pop,
        headline=headline,
        sit_out_reason=sit_out,
        fetched_at=scan.fetched_at,
    )
