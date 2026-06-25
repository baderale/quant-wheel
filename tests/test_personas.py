"""Guard: the council is anonymized — no real-person names anywhere.

The five members are archetypes (Statistician, Value Investor, Risk Manager,
Macro Strategist, Systematic Quant). This test fails loudly if a real name ever
creeps back into a persona's display name or system prompt.
"""

from __future__ import annotations

from quantwheel.agents.personas import (
    buffett,
    burry,
    dalio,
    simons,
    statistician,
)

_MODULES = [statistician, buffett, burry, dalio, simons]

_FORBIDDEN = ("Buffett", "Burry", "Dalio", "Simons", "Renaissance", "Warren", "Michael")

_EXPECTED_NAMES = {
    "The Statistician", "The Value Investor", "The Risk Manager",
    "The Macro Strategist", "The Systematic Quant",
}


def test_no_real_names_in_persona_modules():
    for mod in _MODULES:
        blob = f"{mod.NAME}\n{mod.SYSTEM_PROMPT}"
        for bad in _FORBIDDEN:
            assert bad not in blob, f"{mod.__name__} still references {bad!r}"


def test_persona_names_are_archetypes():
    names = {mod.NAME for mod in _MODULES}
    assert names == _EXPECTED_NAMES


def test_rubric_scales_with_account():
    from quantwheel.agents.personas.base import _rubric

    one = _rubric(1_000.0, 1)
    many = _rubric(25_000.0, 3)
    assert "$1,000" in one and "ONE position" in one
    assert "$25,000" in many and "up to 3 positions" in many
