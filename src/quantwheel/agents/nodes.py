"""LangGraph node functions for the weekly wheel agent.

Each node is a plain function ``(state) -> partial_state_update``. They are bound
to their dependencies (config, llm) in :mod:`quantwheel.agents.graph` via
``functools.partial`` so they stay easy to read and unit-test in isolation.

The state is a ``TypedDict`` of the project's own dataclasses — the same contract
the rest of the system speaks.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, TypedDict

from quantwheel.agents.council import run_council
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import (
    CouncilVerdict,
    ScanResult,
    ScoredCandidate,
    TradeCard,
)
from quantwheel.services.scan_service import run_weekly_scan
from quantwheel.universe import DEFAULT_UNIVERSE


class WheelState(TypedDict, total=False):
    """Shared state threaded through the weekly-agent graph."""

    universe: list[str]
    scan: ScanResult
    verdicts: list[CouncilVerdict]
    pick: Optional[ScoredCandidate]
    pick_verdict: Optional[CouncilVerdict]
    card: Optional[TradeCard]
    approved: bool
    message: str


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def load_universe(state: WheelState) -> WheelState:
    """Resolve the watchlist (explicit, else the small-account shortlist)."""
    return {"universe": state.get("universe") or DEFAULT_UNIVERSE}


def screen(state: WheelState, *, wheel_config: WheelConfig) -> WheelState:
    """Run the quant scan across the universe.

    Note: the keyword is ``wheel_config`` (not ``config``) because LangGraph
    reserves ``config`` for its own ``RunnableConfig`` injection.
    """
    scan = run_weekly_scan(state["universe"], wheel_config)
    return {"scan": scan, "message": f"Scored {len(scan.candidates)} names."}


def council(state: WheelState, *, llm=None, provider: Optional[str] = None,
            model: Optional[str] = None, wheel_config: WheelConfig = DEFAULT) -> WheelState:
    """Convene the council on the gated/enterable candidates only.

    If nothing passed the gates, we skip the (possibly paid) LLM calls entirely.
    """
    enterable = state["scan"].enterable
    if not enterable:
        return {"verdicts": [], "message": "No gated candidates — council skipped."}
    verdicts = run_council(enterable, llm=llm, provider=provider, model=model,
                           config=wheel_config)
    return {"verdicts": verdicts}


def rank_pick(state: WheelState) -> WheelState:
    """Pick the single best name the council is net-favorable on (or sit out)."""
    verdicts = state.get("verdicts") or []
    favorable = [v for v in verdicts if v.aggregate_score > 0]
    if not favorable:
        return {"pick": None, "pick_verdict": None,
                "message": "Council not favorable on any name — sit out this week."}
    best = favorable[0]  # run_council already sorted best-first
    by_ticker = {c.ticker: c for c in state["scan"].candidates}
    return {"pick": by_ticker.get(best.ticker), "pick_verdict": best}


def draft_card(state: WheelState, *, wheel_config: WheelConfig) -> WheelState:
    """Compose the human-facing trade card from the pick + council consensus."""
    pick, verdict = state.get("pick"), state.get("pick_verdict")
    if pick is None:
        return {"card": None}
    return {"card": make_trade_card(pick, verdict, wheel_config)}


def write_record(state: WheelState, *, reports_dir: str = "reports") -> WheelState:
    """Persist the approved card to a dated markdown report (the audit trail).

    Tracker/vault sync is layered on in Phase 5; this guarantees the agent always
    leaves a durable, human-readable record of what was approved.
    """
    import os

    card = state.get("card")
    if card is None:
        return {"message": "Nothing to record."}
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f"{card.as_of}_{card.ticker}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_card_markdown(card, state.get("pick_verdict")))
    return {"message": f"Approved — recorded to {path}."}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_trade_card(
    c: ScoredCandidate, verdict: Optional[CouncilVerdict], config: WheelConfig
) -> TradeCard:
    """Build a :class:`TradeCard` with explicit exit and assignment plans."""
    q, p = c.quote, c.pricing
    take = config.profit_take_pct * p.credit
    cost_basis = q.strike - p.premium
    exit_plan = (
        f"Buy to close at ~{config.profit_take_pct:.0%} of max profit "
        f"(~${take:.0f}); otherwise manage into expiry {q.expiry}."
    )
    assignment_plan = (
        f"If assigned 100 sh @ ${q.strike:.2f} (cost basis ~${cost_basis:.2f}), "
        f"switch to selling weekly covered calls at/above cost basis (~30 delta) "
        f"until called away, then restart the put leg."
    )
    return TradeCard(
        ticker=c.ticker, as_of=c.as_of, action="Sell cash-secured put",
        strike=q.strike, expiry=q.expiry, dte=q.dte, target_credit=p.credit,
        delta=q.delta or 0.0, pop=p.pop, expected_value=p.expected_value,
        breakeven=p.breakeven, contracts=c.sizing.contracts,
        collateral=c.sizing.collateral_used or q.strike * 100,
        exit_plan=exit_plan, assignment_plan=assignment_plan,
        council_consensus=(verdict.consensus if verdict else "n/a"),
        rationale=(f"The model rates {c.ticker} a '{c.verdict.value}' setup "
                   f"(score {c.score:.1f}/6): about a {p.pop:.0%} chance of profit, "
                   f"expected value ~${p.expected_value:.1f} per contract, and "
                   f"roughly a {c.monte_carlo.p_assign:.0%} chance of being assigned "
                   f"the shares."),
    )


def render_card_markdown(card: TradeCard, verdict: Optional[CouncilVerdict]) -> str:
    """Render a trade card as a vault-friendly markdown note."""
    est = ""
    lines = [
        f"# Wheel Trade Card — {card.ticker} ({card.as_of})",
        "",
        "> Educational, not advice. **Verify the live chain in Robinhood before trading.**",
        "> No order is placed by software — this is a suggestion you approve and execute.",
        "",
        f"- **Action:** {card.action}",
        f"- **Strike / Expiry:** ${card.strike:.2f} put · {card.expiry} ({card.dte} DTE)",
        f"- **Target credit:** ${card.target_credit:.0f} / contract  ×{card.contracts}",
        f"- **Delta:** {card.delta:.3f}  ·  **POP:** {card.pop:.0%}  ·  "
        f"**EV:** ${card.expected_value:.1f}",
        f"- **Break-even:** ${card.breakeven:.2f}  ·  **Collateral:** ${card.collateral:,.0f}",
        "",
        f"**Exit plan:** {card.exit_plan}",
        "",
        f"**Assignment plan:** {card.assignment_plan}",
        "",
        f"**Why:** {card.rationale}",
        "",
        "## Council",
        card.council_consensus,
    ]
    if verdict:
        lines.append("")
        for v in verdict.votes:
            lines.append(f"- **{v.persona}** — {v.stance.value} ({v.conviction:.0%}): {v.rationale}")
    return "\n".join(lines) + est + "\n"
