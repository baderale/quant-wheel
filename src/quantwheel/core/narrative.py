"""Plain-language explanations of a trade and the council's reasoning.

The quant core speaks in acronyms — POP, EV, P(assign), delta, DTE. A beginner or
intermediate investor shouldn't have to. These builders turn a sized
:class:`~quantwheel.core.types.Position` and its :class:`TradeCard` into short,
readable paragraphs that define each term inline the first time it appears.

Pure, standard-library text generation — no UI, no LLM, no real-people names.
The dashboard Report tab renders whatever these return.
"""

from __future__ import annotations

from typing import Optional

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import CouncilVerdict, Position, TradeCard

# Beginner-friendly definitions, surfaced in a "what do these terms mean?" panel.
GLOSSARY: dict[str, str] = {
    "Cash-secured put": "You agree to buy 100 shares at the strike price and set "
        "aside the cash to do it. You're paid a premium up front for that promise.",
    "Premium": "The cash you collect for selling the option — your maximum profit "
        "on the trade.",
    "Strike": "The price at which you'd buy the shares if the option is assigned.",
    "Break-even": "The share price where you'd neither make nor lose money — the "
        "strike minus the premium you collected per share.",
    "Probability of profit (POP)": "The model's estimate of the chance the trade "
        "finishes making money.",
    "Assignment": "When the stock closes below the strike at expiry and you're "
        "required to buy the 100 shares at the strike.",
    "Expected value (EV)": "The average result if you made this same bet many "
        "times. A positive number means the odds are in your favor.",
    "Collateral": "The cash set aside to buy the shares if assigned — the strike "
        "times 100 for each contract.",
    "Days to expiry (DTE)": "How many days until the option expires.",
    "Delta": "Roughly the chance the option ends in-the-money; a 0.20 delta means "
        "about a 20% chance of assignment.",
    "Covered call": "Once you own the shares, selling someone the right to buy "
        "them from you at a higher price — collecting more premium while you wait.",
}


# Plain-English meaning of each machine Verdict, keyed by the enum's value.
# ENTER_* are tradeable; WATCH is on-the-radar; SKIP_* records *why* a name was
# rejected so the dashboard explains it instead of silently dropping the ticker.
VERDICT_LEGEND: dict[str, str] = {
    "ENTER_STRONG": "Best setups — cleared every safety check with a strong "
        "technical score. The top names to sell a put on.",
    "ENTER": "A solid trade — cleared every safety check with a good score.",
    "WATCH": "On the radar, not a buy yet — it passed the checks but the score is "
        "only moderate, or the assignment risk is high enough to wait.",
    "SKIP": "Passed the hard gates but the setup is weak — low score, or the "
        "expected value isn't positive.",
    "SKIP_REGIME": "Skipped — the market regime (trend/volatility backdrop) looks "
        "unfavorable for selling puts on this name right now.",
    "SKIP_EARNINGS": "Skipped — an earnings report lands on or before the option's "
        "expiry, which can trigger a big surprise move.",
    "SKIP_LIQUIDITY": "Skipped — the option market is too thin or the bid/ask spread "
        "too wide to expect a fair fill.",
    "SKIP_CAPITAL": "Skipped — the collateral it needs (strike × 100) is more than "
        "your account can deploy, so it doesn't fit.",
}


def _expiry_str(card: TradeCard) -> str:
    return card.expiry.strftime("%b %d, %Y")


def _lean_label(score: float) -> str:
    """Council aggregate score → plain phrase (mirrors the quant labels)."""
    if score >= 0.5:
        return "strongly in favor"
    if score >= 0.15:
        return "cautiously in favor"
    if score > -0.15:
        return "split / neutral"
    if score > -0.5:
        return "leaning against"
    return "strongly against"


def summary_line(position: Position, card: TradeCard) -> str:
    """A one-line headline for the trade, no jargon."""
    n = position.contracts
    s = "s" if n != 1 else ""
    return (
        f"Sell {n} put{s} on {card.ticker} at ${card.strike:.2f} — collect "
        f"about ${position.credit:,.0f}, with roughly a {card.pop:.0%} chance "
        f"of keeping it all."
    )


def explain_trade(position: Position, card: TradeCard) -> str:
    n = position.contracts
    s = "s" if n != 1 else ""
    shares = n * 100
    when = _expiry_str(card)
    return (
        f"**Sell {n} cash-secured put{s} on {card.ticker} at the ${card.strike:.2f} "
        f"strike, expiring {when} ({card.dte} days away).** You collect about "
        f"**${position.credit:,.0f}** today — the *premium*, which is the most you "
        f"can make on this trade. In exchange you set aside "
        f"**${position.collateral:,.0f}** as *collateral* and agree to buy {shares} "
        f"shares of {card.ticker} at ${card.strike:.2f} if it closes below that "
        f"price on {when}."
    )


def explain_odds(position: Position, card: TradeCard) -> str:
    p_assign = position.candidate.monte_carlo.p_assign
    ev_total = card.expected_value * position.contracts
    return (
        f"The model gives this about a **{card.pop:.0%} chance of finishing "
        f"profitable** (its *probability of profit*) — that is, {card.ticker} "
        f"staying above your *break-even* of ${card.breakeven:.2f}, which is the "
        f"strike minus the premium you collected per share. It estimates roughly a "
        f"**{p_assign:.0%}** chance you actually get *assigned* the shares. After "
        f"accounting for that risk, the trade's *expected value* is about "
        f"**${ev_total:,.0f}** — a positive number means the odds tilt your way "
        f"over many repeats."
    )


def explain_exit(card: TradeCard, config: WheelConfig = DEFAULT) -> str:
    take_each = config.profit_take_pct * card.target_credit
    return (
        f"**When to take profit:** you don't have to wait until {_expiry_str(card)}. "
        f"A common plan is to buy the option back once it has lost about "
        f"{config.profit_take_pct:.0%} of its value (around ${take_each:.0f} per "
        f"contract to close), pocket the rest, and free up the cash for the next "
        f"trade."
    )


def explain_assignment(position: Position, card: TradeCard) -> str:
    shares = position.contracts * 100
    return (
        f"**If it goes against you:** should {card.ticker} close below "
        f"${card.strike:.2f}, you'll buy {shares} shares at ${card.strike:.2f}. "
        f"Because you already kept the premium, your effective cost is about "
        f"**${card.breakeven:.2f}** per share. From there the Wheel continues — you "
        f"sell *covered calls* (collecting more premium for agreeing to sell those "
        f"shares a little higher) until they're called away, then start over."
    )


def explain_council(verdict: Optional[CouncilVerdict]) -> str:
    """Readable synthesis of the council vote, using the archetype member names."""
    if verdict is None or not verdict.votes:
        return ""
    votes = verdict.votes
    n = len(votes)
    for_votes = [v for v in votes if v.stance.score > 0]
    against_votes = [v for v in votes if v.stance.score < 0]
    parts = [
        f"The council was **{_lean_label(verdict.aggregate_score)}** on this trade — "
        f"{len(for_votes)} of {n} members in favor."
    ]
    if for_votes:
        top = max(for_votes, key=lambda v: v.conviction)
        parts.append(f"The {top.persona} was most supportive: {top.rationale}")
    if against_votes:
        dissent = max(against_votes, key=lambda v: v.conviction)
        parts.append(f"The main holdout was the {dissent.persona}: {dissent.rationale}")
    elif not for_votes:
        parts.append("Members were split, with no strong conviction either way.")
    return " ".join(parts)


def build_explanation(
    position: Position, card: TradeCard, config: WheelConfig = DEFAULT
) -> list[str]:
    """The ordered plain-English paragraphs shown under a recommended trade."""
    return [
        explain_trade(position, card),
        explain_odds(position, card),
        explain_exit(card, config),
        explain_assignment(position, card),
    ]
