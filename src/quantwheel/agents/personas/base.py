"""The shared scaffold every council persona is built on.

A persona is *one LangChain chain*: a prompt (its identity + the trade evidence)
piped into a chat model that is forced to return a **structured** opinion.
Subclasses change nothing but the system prompt — that's the whole point of the
council, and a clean demonstration of how little code a LangChain agent needs.

The model is **provider-agnostic**: it comes from :func:`quantwheel.agents.llm.make_chat_model`,
so the same persona runs on Claude, a local Ollama model, OpenAI, etc., chosen by
environment variables. The model is also injectable (`llm=`) so the council can be
unit-tested offline with a fake runnable.

LangChain concepts demonstrated here (see ``notebooks/01_learn_langchain.ipynb``):
* ``ChatPromptTemplate`` — system + human messages with variables.
* ``.with_structured_output(PydanticModel)`` — typed, parsed output, no regex.
* LCEL piping — ``prompt | model``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from quantwheel.core.config import DEFAULT
from quantwheel.core.types import CouncilVote, ScoredCandidate, Stance


def _rubric(account: float, max_positions: int) -> str:
    """The shared scoring rubric, parameterised by the actual account context.

    Every persona answers the same question in the same shape, differing only in
    *judgement*. The account size and position count are injected so the council
    reasons about the user's real situation — not a hardcoded $1,000 / one-name
    account — which materially changes the diversification argument.
    """
    if max_positions <= 1:
        size_line = (
            f"a ${account:,.0f} account that holds ONE position at a time "
            f"(no diversification to fall back on)"
        )
        risk_line = (
            "Remember: with a single concentrated position, assignment in a "
            "falling stock can wipe out many weeks of premium and there is no "
            "other holding to soften it."
        )
    else:
        size_line = (
            f"a ${account:,.0f} account that holds up to {max_positions} "
            f"positions at a time"
        )
        risk_line = (
            f"Remember: risk is spread across up to {max_positions} positions, "
            f"but each name can still be assigned in a falling stock — judge "
            f"THIS trade on its own merits and its weight in the portfolio."
        )
    return f"""

You are sitting on an investment council evaluating a single cash-secured PUT
trade (the Wheel strategy) for {size_line}. Selling a cash-secured put means:
collect premium now; if the stock closes below the strike at expiry you are
assigned 100 shares at the strike.

Judge ONLY this trade, in character. Return:
- stance: STRONG_FOR / FOR / NEUTRAL / AGAINST / STRONG_AGAINST
- conviction: 0.0-1.0 (how strongly you hold this stance)
- rationale: 1-3 sentences, in your voice
- key_points: 2-4 short bullet phrases driving your view

{risk_line}
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
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        account: float = DEFAULT.account,
        max_positions: int = DEFAULT.max_positions,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self._llm = llm                 # injected runnable (tests) or None (live)
        self._provider = provider       # None -> from env (QUANTWHEEL_PROVIDER)
        self._model = model             # None -> provider default
        self._temperature = temperature
        self._account = account         # injected so the rubric reflects reality
        self._max_positions = max_positions

    def _runnable(self):
        """Build the LCEL chain: prompt | structured chat model (or injected llm)."""
        from langchain_core.prompts import ChatPromptTemplate

        rubric = _rubric(self._account, self._max_positions)
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt + rubric), ("human", _HUMAN)]
        )
        llm = self._llm
        if llm is None:  # live path — provider chosen by env / args
            from quantwheel.agents.llm import make_chat_model

            llm = make_chat_model(
                provider=self._provider, model=self._model,
                temperature=self._temperature,
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
