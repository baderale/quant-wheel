"""Markov regime model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantwheel.core.models.regime import (
    STATES,
    analyze_regime,
    classify_series,
    fit_transition_matrix,
    project,
)
from quantwheel.core.types import RegimeState


def test_transition_matrix_rows_are_stochastic():
    states = pd.Series([s for s in STATES] * 25)  # cycle through all 4 states
    P = fit_transition_matrix(states, alpha=1.0)
    assert P.shape == (4, 4)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()


def test_projection_preserves_stochasticity():
    states = pd.Series([s for s in STATES] * 25)
    P = fit_transition_matrix(states)
    P5 = project(P, 5)
    assert np.allclose(P5.sum(axis=1), 1.0)


def test_laplace_smoothing_removes_zeros():
    # Only one transition observed; smoothing must still give a full matrix.
    states = pd.Series([RegimeState.RANGE, RegimeState.RANGE])
    P = fit_transition_matrix(states, alpha=1.0)
    assert (P > 0).all()


def test_classify_and_analyze_on_synthetic(synth_bars):
    labels = classify_series(synth_bars).dropna()
    assert set(labels.unique()).issubset(set(STATES))

    res = analyze_regime(synth_bars, horizon_bars=5)
    assert 0.0 <= res.p_favorable <= 1.0
    assert abs(sum(res.horizon_distribution.values()) - 1.0) < 1e-9
    assert res.current_state in STATES
    assert res.expected_dwell_days >= 1.0
