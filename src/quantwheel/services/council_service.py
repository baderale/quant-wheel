"""Council service — facade for the LangChain persona council.

Takes scored candidates from a scan and returns the council's ranked verdicts.
"""

from __future__ import annotations

from typing import Optional

from quantwheel.agents.council import run_council as _run
from quantwheel.core.types import CouncilVerdict, ScoredCandidate


def run_council(
    candidates: list[ScoredCandidate],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> list[CouncilVerdict]:
    """Have the five personas vote on each candidate, then aggregate.

    The LLM provider is configurable (local Ollama, Anthropic, OpenAI, …) via
    ``QUANTWHEEL_PROVIDER``/``QUANTWHEEL_MODEL`` or the ``provider``/``model``
    arguments — see :mod:`quantwheel.agents.llm`.

    Parameters
    ----------
    candidates:
        The scored candidates to deliberate over (usually the gated/enterable
        subset of a :class:`ScanResult`).
    provider, model:
        Override the configured LLM provider / model id.

    Returns
    -------
    list[CouncilVerdict]
        One verdict per candidate, sorted by aggregate council score.
    """
    return _run(candidates, provider=provider, model=model)
