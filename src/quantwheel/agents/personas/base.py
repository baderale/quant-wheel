"""The shared scaffold every council persona is built on.

A persona is *one LangChain chain*: a prompt (its identity + the trade evidence)
piped into a Claude model that is forced to return a **structured** opinion.
Subclasses change nothing but the system prompt — that's the whole point of the
council, and a clean demonstration of how little code a LangChain agent needs.

The model is injectable (`llm=`), so the council can be unit-tested offline with
a fake runnable and run live with Claude in production.

LangChain concepts demonstrated here (see ``notebooks/01_learn_langchain.ipynb``):
* ``ChatPromptTemplate`` — system + human messages with variables.
* ``.with_structured_output(PydanticModel)`` — typed, parsed output, no regex.
* LCEL piping — ``prompt | model``.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field

from quantwheel.core.types import CouncilVote, ScoredCandidate, Stance

DEFAULT_MODEL = os.getenv("QUANTWHEEL_MODEL", "claude-sonnet-4-6")

# Rubric appended to every persona's system prompt so all voters answer the same
# question in the same shape, differing only in *judgement*.
_RUBRIC = """

You are sitting on an investment council evaluating a single cash-secured PUT
trade (the Wheel strategy) for a SMALL $1,000 account that holds ONE position at
a time. Selling a cash-secured put means: collect premium now; if the stock
closes below the strike at expiry you are assigned 100 shares at the strike.

Judge ONLY this trade, in character. Return:
- stance: STRONG_FOR / FOR / NEUTRAL / AGAINST / STRONG_AGAINST
- conviction: 0.0-1.0 (how strongly you hold this stance)
- rationale: 1-3 sentences, in your voice
- key_points: 2-4 short bullet phrases driving your view

Remember the small-account reality: assignment in a falling stock can wipe out
many weeks of premium, and there is no diversification to fall back on.
"""

_HUMAN = "Evaluate this cash-secured put candidate:\n\n{evidence}\n\nGive your verdict."


class PersonaOpinion(BaseModel):
    """The structured shape we force the LLM to return (parsed automatically)."""

    stance: Literal["STRONG_FOR", "FOR", "NEUTRAL", "AGAINST", "STRONG_AGAINST"]
    conviction: float = Field(ge=0.0, le=1.0)
    rationale: str
    key_points: list[str] = Field(default_factory=list)


def format_evidence(c: ScoredCandidate) -> str:
    """Render a candidate into the compact evidence block shown to every persona."""
    q, p, r, mc = c.quote, c.pricing, c.regime, c.monte_carlo
    earn = c.next_earnings.isoformat() if c.next_earnings else "unknown"
    est = " (ESTIMATED quote — verify live)" if q.is_estimated else ""
    return (
        f"Ticker: {c.ticker}  spot ${c.spot:.2f}  (as of {c.as_of})\n"
        f"Machine verdict: {c.verdict.value}  score {c.score:.2f}/6\n"
        f"Trade: sell ${q.strike:.2f} put, {q.dte} DTE, ~{abs(q.delta or 0):.2f} delta{est}\n"
        f"Credit ${p.credit:.0f}/contract  break-even ${p.breakeven:.2f}  "
        f"annualized RoR {p.annualized_ror:.0%}\n"
        f"POP {p.pop:.0%}  expected value ${p.expected_value:.1f}  "
        f"P(assign) {mc.p_assign:.0%}  P(touch) {mc.p_touch:.0%}\n"
        f"Regime: {r.current_state.value}  P(favorable@expiry) {r.p_favorable:.0%}  "
        f"stability {r.stability:.2f}\n"
        f"Technicals: RSI {c.indicators['rsi']:.0f}  BB% {c.indicators['bb_pct']:.2f}  "
        f"realized vol {c.indicators['realized_vol']:.0%}\n"
        f"Next earnings: {earn}\n"
        f"Sizing: {c.sizing.reason}"
    )


class PersonaAgent:
    """One council member. Identity lives entirely in ``system_prompt``."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        *,
        llm=None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self._llm = llm                 # injected runnable (tests) or None (live)
        self._model = model or DEFAULT_MODEL
        self._temperature = temperature

    def _runnable(self):
        """Build the LCEL chain: prompt | structured-Claude (or injected llm)."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt + _RUBRIC), ("human", _HUMAN)]
        )
        llm = self._llm
        if llm is None:  # live path
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                model=self._model, temperature=self._temperature
            ).with_structured_output(PersonaOpinion)
        return prompt | llm

    def evaluate(self, candidate: ScoredCandidate) -> CouncilVote:
        """Return this persona's structured vote on one candidate."""
        opinion: PersonaOpinion = self._runnable().invoke(
            {"evidence": format_evidence(candidate)}
        )
        return CouncilVote(
            persona=self.name,
            ticker=candidate.ticker,
            stance=Stance(opinion.stance),
            conviction=float(opinion.conviction),
            rationale=opinion.rationale,
            key_points=list(opinion.key_points),
        )
