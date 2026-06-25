"""The watchlist of tickers to scan.

A baked-in default (the vault's small-account shortlist) plus a CSV loader so the
list can live in ``scans/shortlist.csv`` or come from the Obsidian vault.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Mirrors the vault's Candidate Shortlist (Tier A first). Used when no CSV is given.
DEFAULT_UNIVERSE: list[str] = ["F", "SOFI", "T", "AAL", "NIO", "MARA", "CLSK"]


def load_shortlist(path: str | Path) -> list[str]:
    """Read tickers from a shortlist CSV (expects a ``ticker`` column)."""
    df = pd.read_csv(path)
    return [str(t).strip().upper() for t in df["ticker"].dropna()]
