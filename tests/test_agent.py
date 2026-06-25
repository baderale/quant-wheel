"""Weekly-agent tests — run offline (scan monkeypatched, council faked)."""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.runnables import RunnableLambda

from quantwheel.agents import graph, nodes
from quantwheel.agents.personas.base import PersonaOpinion
from quantwheel.core.config import DEFAULT
from quantwheel.core.types import ScanResult


def _fake_for():
    op = PersonaOpinion(stance="FOR", conviction=0.8, rationale="liquid, +EV",
                        key_points=["EV>0"])
    return RunnableLambda(lambda _: op)


def test_make_card_has_exit_and_assignment_plans(scored_candidate):
    card = nodes.make_trade_card(scored_candidate, None, DEFAULT)
    assert card.ticker == "NIO"
    assert "close" in card.exit_plan.lower()
    assert "assigned" in card.assignment_plan.lower()
    md = nodes.render_card_markdown(card, None)
    assert "Wheel Trade Card" in md and "NIO" in md


def test_routing_helpers():
    assert graph._route_after_pick({"pick": object()}) == "draft_card"
    assert graph._route_after_pick({"pick": None}) != "draft_card"
    assert graph._route_after_approval({"approved": True}) == "write_record"
    assert graph._route_after_approval({"approved": False}) != "write_record"


def test_full_agent_runs_and_records(monkeypatch, tmp_path, scored_candidate):
    # Stub the (networked) scan with a one-candidate enterable result.
    scan = ScanResult(as_of=date(2026, 6, 25), account=1000.0,
                      universe=["NIO"], candidates=[scored_candidate])
    monkeypatch.setattr(nodes, "run_weekly_scan", lambda *a, **k: scan)
    monkeypatch.chdir(tmp_path)  # write_record -> tmp reports/

    out = graph.run_weekly_agent(
        ["NIO"], llm=_fake_for(), auto_approve=True, thread_id="test-approve",
    )
    assert out["card"] is not None
    assert "recorded" in out["message"].lower()
    assert (tmp_path / "reports").exists()


def test_agent_sits_out_when_council_against(monkeypatch, scored_candidate):
    scan = ScanResult(as_of=date(2026, 6, 25), account=1000.0,
                      universe=["NIO"], candidates=[scored_candidate])
    monkeypatch.setattr(nodes, "run_weekly_scan", lambda *a, **k: scan)
    against = RunnableLambda(lambda _: PersonaOpinion(
        stance="STRONG_AGAINST", conviction=0.9, rationale="too risky", key_points=["x"]))

    out = graph.run_weekly_agent(["NIO"], llm=against, thread_id="test-sitout")
    assert out.get("pick") is None
    assert "sit out" in out["message"].lower()
