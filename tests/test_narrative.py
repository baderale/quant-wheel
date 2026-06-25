"""Plain-language narrative builders — readable, jargon-defined, no real names."""

from __future__ import annotations

from quantwheel.agents.nodes import make_trade_card
from quantwheel.core import narrative
from quantwheel.core.config import DEFAULT
from quantwheel.core.types import (
    CouncilVerdict,
    CouncilVote,
    Position,
    Stance,
)


def _position(candidate):
    p = candidate.pricing
    return Position(candidate=candidate, contracts=1,
                    collateral=candidate.quote.strike * 100, credit=p.credit)


def test_trade_and_odds_are_readable(scored_candidate):
    pos = _position(scored_candidate)
    card = make_trade_card(scored_candidate, None, DEFAULT)

    trade = narrative.explain_trade(pos, card)
    odds = narrative.explain_odds(pos, card)
    assert scored_candidate.ticker in trade
    assert "premium" in trade.lower() and "collateral" in trade.lower()
    # Acronyms are expanded, not left bare.
    assert "probability of profit" in odds.lower()
    assert "expected value" in odds.lower()


def test_explanation_bundle_has_four_paragraphs(scored_candidate):
    pos = _position(scored_candidate)
    card = make_trade_card(scored_candidate, None, DEFAULT)
    paras = narrative.build_explanation(pos, card, DEFAULT)
    assert len(paras) == 4
    assert all(isinstance(p, str) and p for p in paras)


def test_council_explanation_uses_archetype_names(scored_candidate):
    votes = [
        CouncilVote("The Value Investor", "NIO", Stance.FOR, 0.8, "happy to own it"),
        CouncilVote("The Risk Manager", "NIO", Stance.AGAINST, 0.6, "balance sheet thin"),
    ]
    verdict = CouncilVerdict(ticker="NIO", votes=votes, aggregate_score=0.2,
                             consensus="x", quant_score=4.5)
    text = narrative.explain_council(verdict)
    assert "The Value Investor" in text
    assert "The Risk Manager" in text
    for bad in ("Buffett", "Burry", "Dalio", "Simons"):
        assert bad not in text


def test_glossary_is_nonempty():
    assert narrative.GLOSSARY
    assert "Premium" in narrative.GLOSSARY
