"""Live equity screener — find affordable, liquid names market-wide (Yahoo).

This powers *discovery* (reverse search): when nothing on the user's watchlist
fits their account, search the whole US market for names whose strikes their cash
can actually secure. We lean on Yahoo's own screener (via ``yfinance.screen``) as
a cheap *affordability prefilter* — bounding price and dollar-volume server-side —
then hand the survivors to the normal scan, which verifies optionability and runs
the full math. That keeps discovery fast: we only pay for a deep scan on names
that are already in range.

Network-bound, like the rest of ``core/data``. Returns plain ticker strings and
degrades to ``[]`` on any screener error (it's a best-effort convenience).
"""

from __future__ import annotations

from quantwheel.core.config import DEFAULT, WheelConfig

# A cash-secured put at the ~target-delta strike sits near 80% of spot, and must
# satisfy ``strike * 100 <= per-position budget``. So a name is affordable when
# its spot is roughly below ``budget / 80``. A floor avoids sub-$2 penny names.
_PRICE_FLOOR = 2.0
_MIN_DAY_VOLUME = 1_000_000   # liquidity proxy; the option-liquidity gate is later


def affordable_max_price(config: WheelConfig) -> float:
    """Highest spot price whose ~target-delta put could fit one position's budget."""
    budget = config.deployable_cash / max(1, config.max_positions)
    return budget / 80.0


def find_affordable_optionable(
    config: WheelConfig = DEFAULT,
    limit: int = 25,
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """Return up to ``limit`` liquid US tickers priced to fit ``config``.

    Uses Yahoo's screener for a fast, server-side affordability + liquidity
    filter. Optionability is intentionally *not* checked here — the scan that
    follows skips names with no listed options. Returns ``[]`` on any error.
    """
    max_price = affordable_max_price(config)
    if max_price <= _PRICE_FLOOR:
        return []  # account too small to secure any reasonable name

    try:
        import yfinance as yf
        from yfinance import EquityQuery as Q

        query = Q("and", [
            Q("eq", ["region", "us"]),
            Q("gt", ["intradayprice", _PRICE_FLOOR]),
            Q("lt", ["intradayprice", max_price]),
            Q("gt", ["dayvolume", _MIN_DAY_VOLUME]),
        ])
        # Over-fetch so exclusions and duplicates still leave a full batch.
        res = yf.screen(query, size=max(limit * 2, 50),
                        sortField="dayvolume", sortAsc=False)
    except Exception:
        return []

    quotes = res.get("quotes", []) if isinstance(res, dict) else (res or [])
    excluded = {e.upper() for e in exclude}
    out: list[str] = []
    seen: set[str] = set()
    for q in quotes:
        sym = str(q.get("symbol", "")).upper().strip()
        if not sym or sym in seen or sym in excluded:
            continue
        if q.get("quoteType") not in (None, "EQUITY"):
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= limit:
            break
    return out
