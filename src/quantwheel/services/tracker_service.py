"""Tracker service — the built-in trade tracker (local JSON persistence).

CRUD over :class:`~quantwheel.core.types.TradeRecord`. The dashboard's Tracker
tab is a thin view over these functions. Storage is a single JSON file (path
overridable via ``QUANTWHEEL_TRACKER``) so it's trivial to inspect, back up, or
later mirror into the Obsidian vault ``_Trade Tracker`` note.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Optional

from quantwheel.core.types import TradeCard, TradeRecord

_PATH = Path(os.getenv("QUANTWHEEL_TRACKER", "data/trades.json"))
_DATE_FIELDS = ("open_date", "expiry", "close_date")


def _to_json(rec: TradeRecord) -> dict:
    d = asdict(rec)
    for f in _DATE_FIELDS:
        d[f] = d[f].isoformat() if isinstance(d[f], date) else None
    return d


def _from_json(d: dict) -> TradeRecord:
    d = dict(d)
    for f in _DATE_FIELDS:
        d[f] = date.fromisoformat(d[f]) if d.get(f) else None
    return TradeRecord(**d)


def _load() -> list[TradeRecord]:
    if not _PATH.exists():
        return []
    with _PATH.open(encoding="utf-8") as fh:
        return [_from_json(d) for d in json.load(fh)]


def _save(records: list[TradeRecord]) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    with _PATH.open("w", encoding="utf-8") as fh:
        json.dump([_to_json(r) for r in records], fh, indent=2)


# --------------------------------------------------------------------------- #
# Public API (the facade)
# --------------------------------------------------------------------------- #

def list_trades(status: Optional[str] = None) -> list[TradeRecord]:
    """Return all tracked trades, optionally filtered by status."""
    records = _load()
    return [r for r in records if status is None or r.status == status]


def add_trade(record: TradeRecord) -> TradeRecord:
    """Persist a newly opened trade (assigns an id if missing)."""
    if not record.id:
        record.id = uuid.uuid4().hex[:8]
    records = _load()
    records.append(record)
    _save(records)
    return record


def update_trade(trade_id: str, **fields) -> TradeRecord:
    """Update fields on an existing trade (e.g. mark assigned, advance phase)."""
    records = _load()
    for r in records:
        if r.id == trade_id:
            for k, v in fields.items():
                setattr(r, k, v)
            _save(records)
            return r
    raise KeyError(f"No trade with id {trade_id!r}.")


def close_trade(trade_id: str, debit_close: float, realized_pnl: float) -> TradeRecord:
    """Close a trade and record realized P&L."""
    return update_trade(trade_id, status="CLOSED", debit_close=debit_close,
                        realized_pnl=realized_pnl, close_date=date.today())


def record_from_card(card: TradeCard) -> TradeRecord:
    """Convert an approved :class:`TradeCard` into an open :class:`TradeRecord`."""
    return TradeRecord(
        id="", ticker=card.ticker, leg="CSP", open_date=card.as_of,
        strike=card.strike, expiry=card.expiry, contracts=card.contracts,
        credit_open=card.target_credit, status="OPEN", phase="PUT",
        notes=f"From card: {card.rationale}",
    )


def summary() -> dict[str, float]:
    """Portfolio roll-up for the dashboard header (income, open risk, win rate)."""
    records = _load()
    closed = [r for r in records if r.status == "CLOSED"]
    realized = sum(r.realized_pnl for r in closed)
    open_collateral = sum(r.strike * 100 * r.contracts
                          for r in records if r.status == "OPEN")
    wins = sum(1 for r in closed if r.realized_pnl > 0)
    return {
        "n_open": float(sum(1 for r in records if r.status == "OPEN")),
        "n_closed": float(len(closed)),
        "realized_pnl": float(realized),
        "open_collateral": float(open_collateral),
        "win_rate": float(wins / len(closed)) if closed else 0.0,
    }
