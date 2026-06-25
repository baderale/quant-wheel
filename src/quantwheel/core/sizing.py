"""Position sizing for a one-position small account.

This fixes the old tool's "$25k or bust" assumption, which silently returned
``0 contracts`` for a $1,000 account. Here, when a trade doesn't fit, we say
*why* and what would fit — because at $1,000 the binding constraint isn't
strategy, it's arithmetic: a cash-secured put ties up ``strike * 100`` in cash.

Worked reality for a $1,000 account with a $300 buffer ($700 deployable):

* A $14 stock needs ~$1,300 collateral  -> does **not** fit.
* Only underlyings with a strike below ~$7 fit a true CSP at this size.

That's the concentration trap of a tiny account, stated plainly.
"""

from __future__ import annotations

import math

from quantwheel.core.config import DEFAULT, WheelConfig
from quantwheel.core.types import Sizing


def size_position(required_cash_per_contract: float, config: WheelConfig = DEFAULT) -> Sizing:
    """How many contracts fit, plus a human explanation when the answer is zero.

    Parameters
    ----------
    required_cash_per_contract:
        Collateral for one cash-secured put = ``strike * 100``.
    """
    deployable = config.deployable_cash * config.max_pos_pct
    account, buffer = config.account, config.cash_buffer

    if required_cash_per_contract <= 0:
        return Sizing(account, config.max_pos_pct, 0, 0.0, account, False,
                      "Invalid collateral (<= 0).")

    contracts = int(math.floor(deployable / required_cash_per_contract))

    if contracts < 1:
        max_strike = deployable / 100.0
        reason = (
            f"Does not fit: one contract needs "
            f"${required_cash_per_contract:,.0f} collateral, but only "
            f"${deployable:,.0f} is deployable (${account:,.0f} account "
            f"- ${buffer:,.0f} buffer). At this size a cash-secured put only "
            f"fits underlyings with a strike below ~${max_strike:,.2f}. "
            f"Consider a cheaper name or a put credit spread (Level 3)."
        )
        return Sizing(account, config.max_pos_pct, 0, 0.0, deployable, False, reason)

    collateral_used = contracts * required_cash_per_contract
    buffer_remaining = account - collateral_used
    reason = (
        f"Fits {contracts} contract(s): ${collateral_used:,.0f} collateral, "
        f"${buffer_remaining:,.0f} left idle."
    )
    return Sizing(account, config.max_pos_pct, contracts, collateral_used,
                  buffer_remaining, True, reason)
