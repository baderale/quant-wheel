"""The investment council: five personas deliberate, then we aggregate.

Each persona — the Statistician, the Value Investor, the Risk Manager, the Macro
Strategist and the Systematic Quant — votes on the *same* evidence. We blend their
stances, weighted by each persona's stated conviction, into a single score in
``[-1, +1]`` and a short, deterministic consensus summary.

Why deterministic aggregation (no extra LLM call)? It's free, reproducible, and
testable. The *diversity of judgement* comes from the five personas; the
arithmetic that combines them should be transparent, not another black box.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from quantwheel.agents.personas import buffett, burry, dalio, simons, statistician
from quantwheel.agents.personas.base import PersonaAgent
from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import CouncilVerdict, CouncilVote, ScoredCandidate, Stance


def build_council(
    llm=None, provider: Optional[str] = None, model: Optional[str] = None,
    config: WheelConfig = DEFAULT,
) -> list[PersonaAgent]:
    """Instantiate the five council members.

    ``provider``/``model`` choose the LLM (Ollama, Anthropic, OpenAI, …) — or
    leave them ``None`` to read ``QUANTWHEEL_PROVIDER``/``QUANTWHEEL_MODEL`` from
    the environment. Inject ``llm`` to run fully offline in tests. ``config``
    supplies the account size / position count so each persona's rubric reflects
    the user's real situation instead of a hardcoded small account.
    """
    ctx = {"account": config.account, "max_positions": config.max_positions}
    return [
        statistician.make(llm=llm, provider=provider, model=model, **ctx),
        buffett.make(llm=llm, provider=provider, model=model, **ctx),
        burry.make(llm=llm, provider=provider, model=model, **ctx),
        dalio.make(llm=llm, provider=provider, model=model, **ctx),
        simons.make(llm=llm, provider=provider, model=model, **ctx),
    ]


def _aggregate_score(votes: list[CouncilVote]) -> float:
    """Conviction-weighted mean of stance scores, in [-1, +1]."""
    total_w = sum(v.conviction for v in votes)
    if total_w <= 0:
        return sum(v.stance.score for v in votes) / len(votes) if votes else 0.0
    return sum(v.stance.score * v.conviction for v in votes) / total_w


def _label(score: float) -> str:
    if score >= 0.5:
        return "strongly favorable"
    if score >= 0.15:
        return "cautiously favorable"
    if score > -0.15:
        return "split / neutral"
    if score > -0.5:
        return "cautiously against"
    return "strongly against"


def _consensus(votes: list[CouncilVote], score: float) -> str:
    """Human-readable synthesis: the tally, the net lean, and the loudest dissent."""
    tally = Counter(v.stance.name for v in votes)
    tally_str = ", ".join(f"{n}×{s}" for s, n in tally.most_common())
    # Loudest dissenter = highest-conviction vote on the minority side of the net.
    against = [v for v in votes if v.stance.score < 0] if score >= 0 else \
              [v for v in votes if v.stance.score > 0]
    note = ""
    if against:
        loud = max(against, key=lambda v: v.conviction)
        note = f" Key dissent — {loud.persona}: {loud.rationale}"
    return f"Net {score:+.2f} ({_label(score)}). Votes: {tally_str}.{note}"


def deliberate(candidate: ScoredCandidate, council: list[PersonaAgent]) -> CouncilVerdict:
    """Collect every persona's vote on one candidate and aggregate."""
    votes = [agent.evaluate(candidate) for agent in council]
    score = _aggregate_score(votes)
    return CouncilVerdict(
        ticker=candidate.ticker,
        votes=votes,
        aggregate_score=score,
        consensus=_consensus(votes, score),
        quant_score=candidate.score,
    )


def run_council(
    candidates: list[ScoredCandidate], llm=None,
    provider: Optional[str] = None, model: Optional[str] = None,
    config: WheelConfig = DEFAULT,
) -> list[CouncilVerdict]:
    """Deliberate over every candidate; return verdicts sorted best-first."""
    council = build_council(llm=llm, provider=provider, model=model, config=config)
    verdicts = [deliberate(c, council) for c in candidates]
    verdicts.sort(key=lambda v: -v.aggregate_score)
    return verdicts
