"""Council service — facade for the LangChain persona council.

Takes scored candidates from a scan and returns the council's ranked verdicts.
"""

from __future__ import annotations

from typing import Optional

from quantwheel.agents.council import run_council as _run
from quantwheel.core.types import CouncilVerdict, ScoredCandidate


def run_council(
    candidates: list[ScoredCandidate],
    model: Optional[str] = None,
) -> list[CouncilVerdict]:
    """Have the five personas vote on each candidate, then aggregate.

    Requires ``ANTHROPIC_API_KEY`` in the environment (the personas call Claude).

    Parameters
    ----------
    candidates:
        The scored candidates to deliberate over (usually the gated/enterable
        subset of a :class:`ScanResult`).
    model:
        Claude model id override; defaults to ``QUANTWHEEL_MODEL`` / sonnet.

    Returns
    -------
    list[CouncilVerdict]
        One verdict per candidate, sorted by aggregate council score.
    """
    return _run(candidates, model=model)
