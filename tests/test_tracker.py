"""Trade-tracker persistence tests (isolated to a temp JSON file)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quantwheel.services import tracker_service as ts
from quantwheel.core.types import TradeRecord


@pytest.fixture(autouse=True)
def _temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_PATH", tmp_path / "trades.json")


def _rec(ticker="NIO") -> TradeRecord:
    return TradeRecord(id="", ticker=ticker, leg="CSP", open_date=date(2026, 6, 25),
                       strike=4.5, expiry=date(2026, 7, 2), contracts=1, credit_open=11.0)


def test_add_assigns_id_and_persists():
    rec = ts.add_trade(_rec())
    assert rec.id
    assert len(ts.list_trades()) == 1
    assert ts.list_trades(status="OPEN")[0].ticker == "NIO"


def test_round_trip_dates_survive_serialization():
    ts.add_trade(_rec())
    loaded = ts.list_trades()[0]
    assert loaded.open_date == date(2026, 6, 25)
    assert loaded.expiry == date(2026, 7, 2)


def test_close_trade_records_pnl_and_summary():
    rec = ts.add_trade(_rec())
    ts.close_trade(rec.id, debit_close=5.0, realized_pnl=6.0)
    closed = ts.list_trades(status="CLOSED")
    assert len(closed) == 1 and closed[0].realized_pnl == 6.0
    s = ts.summary()
    assert s["realized_pnl"] == 6.0 and s["win_rate"] == 1.0


def test_update_unknown_id_raises():
    with pytest.raises(KeyError):
        ts.update_trade("nope", status="CLOSED")
