"""Discovery / reverse-search — screener math + service wiring (no network).

The screener itself hits Yahoo, so we monkeypatch it; these tests pin the
affordability math and the service's score-the-survivors behaviour offline.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from quantwheel.core.config import DEFAULT
from quantwheel.core.data import screener
from quantwheel.core.types import ScanResult
from quantwheel.services import discovery_service


def test_affordable_max_price_scales_with_account():
    # $1,000 account, $300 buffer, 1 position -> $700 deployable -> 700/80.
    assert screener.affordable_max_price(DEFAULT) == 700.0 / 80.0
    big = replace(DEFAULT, account=25_000.0, max_positions=5)
    # 24,700 deployable / 5 positions / 80 ≈ $61.75 per-name price ceiling.
    assert round(screener.affordable_max_price(big), 2) == 61.75


def test_discover_returns_empty_when_screener_finds_nothing(monkeypatch):
    monkeypatch.setattr(screener, "find_affordable_optionable",
                        lambda *a, **k: [])
    result = discovery_service.discover(DEFAULT, as_of=date(2026, 6, 25))
    assert isinstance(result, ScanResult)
    assert result.candidates == []


def test_discover_scores_screened_symbols(monkeypatch, scored_candidate):
    monkeypatch.setattr(discovery_service.screener, "find_affordable_optionable",
                        lambda *a, **k: ["NIO"])
    sentinel = ScanResult(as_of=date(2026, 6, 25), account=1000.0,
                          universe=["NIO"], candidates=[scored_candidate])
    monkeypatch.setattr(discovery_service.scoring, "score_universe",
                        lambda *a, **k: sentinel)
    result = discovery_service.discover(DEFAULT, exclude=("F",))
    assert result is sentinel
    assert result.candidates[0].ticker == "NIO"
