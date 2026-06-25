"""Contract/type tests."""

from __future__ import annotations

from datetime import date

from quantwheel.core.types import OptionQuote, Stance


def _quote(bid: float, ask: float, mid: float) -> OptionQuote:
    return OptionQuote(
        ticker="X", expiry=date(2026, 1, 1), dte=7, strike=10.0,
        bid=bid, ask=ask, mid=mid, iv=0.3, open_interest=0, volume=0,
    )


def test_spread_pct_is_nan_without_two_sided_market():
    import math
    assert math.isnan(_quote(0.0, 0.0, 0.5).spread_pct)


def test_spread_pct_computes_with_real_quotes():
    q = _quote(0.90, 1.10, 1.00)
    assert abs(q.spread_pct - 0.20) < 1e-9


def test_quote_defaults_to_live_not_estimated():
    assert _quote(1, 1, 1).is_estimated is False


def test_stance_scores_are_ordered():
    assert Stance.STRONG_FOR.score > Stance.FOR.score > Stance.NEUTRAL.score
    assert Stance.NEUTRAL.score > Stance.AGAINST.score > Stance.STRONG_AGAINST.score
