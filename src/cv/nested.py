"""
Nested Walk-Forward Cross-Validation (P1.3)
===========================================

The Phase-18 overlay parameters (stop-loss floor, s1/s3 switching thresholds,
risk-budget multiplier, policy mix) are currently tuned on the same OOS
folds that the thesis reports on. That is a textbook leakage pattern — the
out-of-sample metric is biased upward because the hyperparameters have
"seen" the test distribution via the tuner's selection rule.

This module provides a leakage-free alternative: **nested walk-forward
CV**. It does not run training itself — it produces the index splits that
an outer driver (``run_walkforward.py`` with ``--nested``) can feed to a
tuner and a final evaluator.

Shape
-----

For each *outer* fold (expanding training window, fixed-length OOS test
window), we carve the outer training window into *inner* folds with the
same expanding scheme. Hyperparameters are tuned only on the inner folds
(inner-train fits model, inner-test scores configurations). The winning
config is then refit on the full outer-train window and scored exactly
once on the outer-test window — that single score is the reported metric.

Diagrams
--------

    Outer fold i (expanding):
       |---- train[0 : t_i] ----|---- test[t_i : t_i + H] ----|

    Inner folds built from train[0 : t_i]:
       |---- in-train[0 : s_j] ----|---- in-test[s_j : s_j + h] ----|
       (j = 1 ... n_inner, with s_j = t_i - n_inner*h + (j-1)*h ... t_i)

No inner split ever touches data with index >= t_i, so the test window of
outer fold i remains untouched by the tuning procedure.

Design notes
------------

* Indices are *integer positions* into a monthly-aligned DatetimeIndex.
  That lets callers use numpy slicing without repeatedly comparing dates.
* ``make_nested_walkforward`` is pure (no I/O); it is trivial to unit-test.
* The outer split matches what ``run_walkforward.py`` already computes
  (expanding-window, ``test_window_months`` per fold) so enabling
  ``--nested`` does not change outer-fold boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class InnerFold:
    """A single inner (tuning) fold carved out of an outer training window."""

    fold_id: int
    train_start: int       # inclusive
    train_end: int         # exclusive
    test_start: int        # inclusive
    test_end: int          # exclusive

    def __post_init__(self) -> None:
        assert self.train_start < self.train_end <= self.test_start < self.test_end


@dataclass
class OuterFold:
    """One outer fold with its inner-CV splits attached."""

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    inner: List[InnerFold] = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.train_start < self.train_end == self.test_start < self.test_end


def make_nested_walkforward(
    n_obs: int,
    oos_start_idx: int,
    test_window: int,
    n_inner: int = 3,
    inner_test_window: int = None,
    min_train: int = 36,
) -> List[OuterFold]:
    """Build nested walk-forward splits.

    Parameters
    ----------
    n_obs : int
        Total number of observations (rows) in the dataset.
    oos_start_idx : int
        Position of the first OOS month. All rows < this index are
        considered available as initial training data.
    test_window : int
        Outer-fold test length (e.g. 24 months).
    n_inner : int
        Number of inner folds per outer fold.
    inner_test_window : int, optional
        Inner-fold test length. Defaults to ``test_window``.
    min_train : int
        Minimum number of training rows required *inside* an inner fold.
        Inner folds whose training segment would be shorter than this are
        dropped — this can reduce ``n_inner`` for the earliest outer fold.

    Returns
    -------
    List[OuterFold] with inner folds populated.

    Raises
    ------
    ValueError
        If the configured windows cannot carve at least one outer fold.
    """
    if inner_test_window is None:
        inner_test_window = test_window
    if test_window <= 0 or inner_test_window <= 0:
        raise ValueError("test windows must be positive")
    if n_inner < 1:
        raise ValueError("n_inner must be >= 1")
    if oos_start_idx <= min_train:
        raise ValueError(
            f"oos_start_idx={oos_start_idx} leaves no room for min_train={min_train}"
        )

    outer_folds: List[OuterFold] = []
    t = oos_start_idx
    fold_id = 0
    while t + test_window <= n_obs:
        outer = OuterFold(
            fold_id=fold_id,
            train_start=0,
            train_end=t,
            test_start=t,
            test_end=t + test_window,
        )

        # Build n_inner inner folds. Each inner test window is immediately
        # before the outer test window to keep the most recent information
        # in the tuning signal. Inner fold j has:
        #   in_test = [t - (n_inner - j) * h - h, t - (n_inner - j) * h)
        #   in_train = [0, in_test.start)
        h = inner_test_window
        for j in range(n_inner):
            in_test_end = t - (n_inner - 1 - j) * h
            in_test_start = in_test_end - h
            in_train_end = in_test_start
            if in_train_end - 0 < min_train or in_test_start < 0:
                continue
            outer.inner.append(
                InnerFold(
                    fold_id=j,
                    train_start=0,
                    train_end=in_train_end,
                    test_start=in_test_start,
                    test_end=in_test_end,
                )
            )

        outer_folds.append(outer)
        t += test_window
        fold_id += 1

    if not outer_folds:
        raise ValueError(
            f"No outer folds: oos_start={oos_start_idx}, test_window={test_window}, "
            f"n_obs={n_obs}"
        )
    return outer_folds


def describe(folds: Sequence[OuterFold]) -> str:
    """Human-readable summary of a nested split — useful for run logs."""
    lines = [f"Nested walk-forward: {len(folds)} outer folds"]
    for o in folds:
        lines.append(
            f"  outer#{o.fold_id}: train=[0,{o.train_end}) "
            f"test=[{o.test_start},{o.test_end})  inner={len(o.inner)}"
        )
        for i in o.inner:
            lines.append(
                f"    inner#{i.fold_id}: train=[{i.train_start},{i.train_end}) "
                f"test=[{i.test_start},{i.test_end})"
            )
    return "\n".join(lines)
