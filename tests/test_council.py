"""Council tests — run fully offline with a fake LLM (no API key needed).

We inject a ``RunnableLambda`` that returns a canned :class:`PersonaOpinion`, so
we exercise the real chain wiring (prompt -> parse -> CouncilVote) and the
aggregation math without calling Claude.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from quantwheel.agents import council
from quantwheel.agents.personas.base import PersonaAgent, PersonaOpinion
from quantwheel.core.types import CouncilVote, Stance


def _fake(stance: str, conviction: float = 0.8):
    """A runnable that ignores the prompt and returns a fixed opinion."""
    op = PersonaOpinion(stance=stance, conviction=conviction,
                        rationale=f"because {stance}", key_points=["a", "b"])
    return RunnableLambda(lambda _: op)


def test_persona_evaluate_parses_to_vote(scored_candidate):
    agent = PersonaAgent("Tester", "You are a tester.", llm=_fake("FOR", 0.7))
    vote = agent.evaluate(scored_candidate)
    assert isinstance(vote, CouncilVote)
    assert vote.persona == "Tester"
    assert vote.stance == Stance.FOR
    assert vote.conviction == 0.7
    assert vote.ticker == "NIO"


def test_deliberation_aggregates_majority_for(scored_candidate):
    agents = [
        PersonaAgent("A", "p", llm=_fake("STRONG_FOR", 0.9)),
        PersonaAgent("B", "p", llm=_fake("FOR", 0.7)),
        PersonaAgent("C", "p", llm=_fake("FOR", 0.6)),
        PersonaAgent("D", "p", llm=_fake("NEUTRAL", 0.5)),
        PersonaAgent("E", "p", llm=_fake("AGAINST", 0.8)),
    ]
    verdict = council.deliberate(scored_candidate, agents)
    assert verdict.aggregate_score > 0          # net favorable
    assert len(verdict.votes) == 5
    assert "Net" in verdict.consensus
    assert "Key dissent" in verdict.consensus   # E dissents


def test_unanimous_against_is_negative(scored_candidate):
    agents = [PersonaAgent(f"P{i}", "p", llm=_fake("STRONG_AGAINST", 0.9))
              for i in range(5)]
    verdict = council.deliberate(scored_candidate, agents)
    assert verdict.aggregate_score < -0.5


def test_run_council_sorts_best_first(scored_candidate):
    # Two candidates, same fake -> equal scores; just assert it returns verdicts.
    verdicts = council.run_council([scored_candidate], llm=_fake("FOR", 0.8))
    assert len(verdicts) == 1
    assert verdicts[0].ticker == "NIO"
