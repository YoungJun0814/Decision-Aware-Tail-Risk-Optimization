"""Regression tests for P1.3 — nested walk-forward CV splits."""

from __future__ import annotations

import pytest

from src.cv import make_nested_walkforward
from src.cv.nested import describe


def test_basic_shape():
    folds = make_nested_walkforward(
        n_obs=240, oos_start_idx=120, test_window=24, n_inner=3
    )
    assert len(folds) == 5  # (240 - 120) / 24
    for o in folds:
        assert o.test_end - o.test_start == 24
        assert o.train_end == o.test_start
        assert o.train_start == 0


def test_no_leakage_inner_never_sees_outer_test():
    """Every inner test index must be strictly less than the outer test start."""
    folds = make_nested_walkforward(
        n_obs=240, oos_start_idx=120, test_window=24, n_inner=3
    )
    for o in folds:
        assert o.inner, "Expected at least one inner fold"
        for i in o.inner:
            assert i.train_end <= i.test_start  # inner itself is leakage-free
            assert i.test_end <= o.test_start   # inner never peeks at outer test
            assert i.train_end <= o.train_end   # inner train is within outer train


def test_min_train_drops_inner_folds_when_needed():
    """With min_train close to oos_start, earlier inner folds drop out."""
    folds = make_nested_walkforward(
        n_obs=240,
        oos_start_idx=60,
        test_window=24,
        n_inner=3,
        inner_test_window=12,
        min_train=30,
    )
    first = folds[0]
    # inner spots would start at positions 60-36, 60-24, 60-12 = 24, 36, 48
    # so only the ones with in_train_end >= 30 survive → drops the first.
    assert len(first.inner) == 2
    for i in first.inner:
        assert i.train_end >= 30


def test_invalid_args():
    with pytest.raises(ValueError):
        make_nested_walkforward(n_obs=100, oos_start_idx=200, test_window=24)
    with pytest.raises(ValueError):
        make_nested_walkforward(n_obs=240, oos_start_idx=120, test_window=0)
    with pytest.raises(ValueError):
        make_nested_walkforward(
            n_obs=240, oos_start_idx=120, test_window=24, n_inner=0
        )


def test_describe_runs():
    folds = make_nested_walkforward(
        n_obs=240, oos_start_idx=120, test_window=24, n_inner=3
    )
    s = describe(folds)
    assert "outer#0" in s
    assert "inner#" in s
