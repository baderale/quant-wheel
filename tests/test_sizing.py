"""Position-sizing tests — the small-account reality and the buffer rule."""

from __future__ import annotations

from dataclasses import replace

from quantwheel.core.config import DEFAULT
from quantwheel.core.sizing import size_position


def test_expensive_name_does_not_fit_1k():
    # F-like: $13 strike -> $1,300 collateral, > $700 deployable.
    s = size_position(1_300.0, DEFAULT)
    assert s.contracts == 0
    assert not s.fits
    assert "does not fit" in s.reason.lower()


def test_cheap_name_fits_one_contract():
    # NIO-like: $4.50 strike -> $450 collateral, < $700 deployable.
    s = size_position(450.0, DEFAULT)
    assert s.contracts == 1
    assert s.fits
    assert s.collateral_used == 450.0


def test_buffer_is_never_breached():
    # Even when several contracts fit, collateral stays within deployable cash.
    cfg = replace(DEFAULT, account=5_000.0, cash_buffer=500.0)
    s = size_position(450.0, cfg)
    assert s.collateral_used <= cfg.deployable_cash * cfg.max_pos_pct
    assert s.buffer_remaining >= 0


def test_invalid_collateral_is_rejected():
    s = size_position(0.0, DEFAULT)
    assert s.contracts == 0 and not s.fits
