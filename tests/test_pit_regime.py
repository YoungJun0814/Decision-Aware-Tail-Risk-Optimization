"""
Regression test for P0.1 — Point-in-Time regime generator.

Two guarantees are tested:

  (1) **PIT-correctness.** The posterior probability emitted for OOS date
      ``t`` must depend only on features with index <= t. We verify this
      by comparing the OOS row produced by the full PIT run against the
      OOS row produced by a PIT run that sees features only up to ``t``.
      Because the per-month refit in ``generate_pit_probs`` is
      deterministic (random_state is fixed inside the hierarchical HMM),
      these two rows must be bit-equal.

  (2) **Loader contract.** ``get_regime_4state(require_pit=True)`` must
      raise FileNotFoundError when the PIT CSV is missing — no silent
      fallback to the full-sample HMM.

These tests use a synthetic features frame so they run fast (< 10 s) and
do not require network access.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_loader


hmmlearn = pytest.importorskip(
    "hmmlearn",
    reason="hmmlearn is required for PIT regime tests",
)


def _synthetic_features(n_months: int = 180, seed: int = 0) -> pd.DataFrame:
    """Two-regime synthetic series: alternating calm/stressed blocks.

    Returns have two means (positive in calm, negative in stressed) with
    different volatility. VIX is proportional to realised volatility.
    Enough separation that the HMM converges consistently.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2007-01-31", periods=n_months, freq="ME")
    # Regime switches roughly every 18 months.
    regime = np.zeros(n_months, dtype=int)
    block = 0
    i = 0
    while i < n_months:
        length = rng.integers(12, 24)
        regime[i:i + length] = block % 2
        i += length
        block += 1

    returns = np.where(
        regime == 0,
        rng.normal(0.012, 0.025, n_months),   # calm
        rng.normal(-0.015, 0.060, n_months),  # stressed
    )
    rolling_vol = pd.Series(returns).rolling(3).std().bfill().values * math.sqrt(12)
    vix = 12 + 80 * rolling_vol + rng.normal(0, 1.5, n_months)

    df = pd.DataFrame(
        {"Returns": returns, "RollingVol": rolling_vol, "VIX": vix},
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_pit_posterior_depends_only_on_past():
    """Refitting on features[:t] must reproduce the OOS row for date t."""
    from src.gen_regime_4state_pit import generate_pit_probs

    features = _synthetic_features(n_months=180, seed=0)
    start_oos = features.index[120]  # last 60 months are OOS

    # Full PIT pass.
    full = generate_pit_probs(
        features,
        start_oos=str(start_oos.date()),
        min_train_months=60,
        smoothing_alpha=0.3,
        verbose=False,
    )

    # Pick three OOS dates to check (beginning, middle, end of OOS).
    oos_dates = full.index[full["source"] == "oos_per_month_refit"]
    picks = [oos_dates[0], oos_dates[len(oos_dates) // 2], oos_dates[-1]]

    prob_cols = ["Prob_Bull", "Prob_Sideways", "Prob_Correction", "Prob_Crisis"]

    for t in picks:
        truncated = features.loc[features.index <= t]
        # Re-run PIT generator on truncated features with the same OOS
        # start but a shorter end. The row for `t` in this truncated run
        # must equal the row for `t` in the full run.
        partial = generate_pit_probs(
            truncated,
            start_oos=str(start_oos.date()),
            min_train_months=60,
            smoothing_alpha=0.3,
            verbose=False,
        )
        assert t in partial.index, (
            f"Truncated PIT run did not emit row for {t.date()}"
        )
        full_row = full.loc[t, prob_cols].astype(float).values
        part_row = partial.loc[t, prob_cols].astype(float).values
        assert np.allclose(full_row, part_row, atol=1e-10), (
            f"PIT violation at {t.date()}: full={full_row} "
            f"vs truncated={part_row}"
        )


def test_require_pit_raises_when_file_missing(tmp_path, monkeypatch):
    """require_pit=True must raise rather than fall back silently."""
    monkeypatch.setattr(
        data_loader, "REGIME_4STATE_PIT_PATH", tmp_path / "does_not_exist.csv"
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        data_loader.get_regime_4state(require_pit=True)
    assert "PIT regime CSV is missing" in str(excinfo.value)
    assert "gen_regime_4state_pit" in str(excinfo.value)


def test_default_loader_warns_on_fallback(tmp_path, monkeypatch, capsys):
    """Default call with only the full-sample CSV present must warn."""
    # Build a tiny full-sample CSV.
    full_path = tmp_path / "regime_4state.csv"
    dates = pd.date_range("2007-01-31", periods=3, freq="ME")
    pd.DataFrame(
        {
            "Prob_Bull": [0.7, 0.6, 0.5],
            "Prob_Sideways": [0.2, 0.2, 0.2],
            "Prob_Correction": [0.07, 0.15, 0.2],
            "Prob_Crisis": [0.03, 0.05, 0.1],
        },
        index=dates,
    ).rename_axis("Date").to_csv(full_path)

    monkeypatch.setattr(
        data_loader, "REGIME_4STATE_PIT_PATH", tmp_path / "missing_pit.csv"
    )
    monkeypatch.setattr(data_loader, "REGIME_4STATE_PATH", full_path)

    df = data_loader.get_regime_4state()  # require_pit default False
    captured = capsys.readouterr().out
    assert not df.empty
    assert "look-ahead bias" in captured.lower()
    assert "require_pit=true" in captured.lower() or "pit" in captured.lower()
