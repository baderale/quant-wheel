"""Report service — assembles the plain-language, multi-position recommendation."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from quantwheel.core.config import DEFAULT
from quantwheel.core.types import GateResult, ScanResult, Verdict
from quantwheel.services.report_service import build_report


def _scan(candidates, account=1000.0):
    return ScanResult(as_of=date(2026, 6, 25), account=account,
                      universe=[c.ticker for c in candidates], candidates=candidates)


def test_report_recommends_a_specific_trade(scored_candidate):
    report = build_report(_scan([scored_candidate]), None, DEFAULT)
    assert len(report.items) == 1
    item = report.items[0]
    assert item.card.ticker == "NIO"
    assert item.card.contracts == item.position.contracts  # card reflects allocation
    assert item.summary and item.explanation
    assert report.total_credit > 0
    assert "NIO" in item.summary


def test_report_caps_at_max_positions(scored_candidate):
    cands = [replace(scored_candidate, ticker=t,
                     quote=replace(scored_candidate.quote, ticker=t, strike=4.5))
             for t in ("F", "SOFI", "NIO")]
    config = replace(DEFAULT, account=25_000.0, max_positions=2)
    report = build_report(_scan(cands, account=25_000.0), None, config)
    assert len(report.items) == 2
    assert report.cash_deployed <= config.deployable_cash + 1e-6


def test_report_sits_out_when_nothing_enterable(scored_candidate):
    blocked = replace(scored_candidate, verdict=Verdict.SKIP_CAPITAL,
                      gates=GateResult(False, True, True, False, ["does not fit"]))
    report = build_report(_scan([blocked]), None, DEFAULT)
    assert report.items == []
    assert report.sit_out_reason
    assert "sit" in report.headline.lower() or "sit" in report.sit_out_reason.lower()
