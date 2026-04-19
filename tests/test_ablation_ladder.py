"""Smoke tests for P2.5 — ablation ladder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ablation_ladder.py"


def _make_asset_returns(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    assets = ["SPY", "AGG", "BIL"]
    data = rng.normal(0.005, 0.03, (180, 3))
    df = pd.DataFrame(data, index=dates, columns=assets)
    df.index.name = "Date"
    p = tmp_path / "asset_returns.csv"
    df.to_csv(p)
    return p


def _make_model_returns(tmp_path: Path, seed: int, name: str) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-07-31", periods=72, freq="ME")
    s = rng.normal(0.008, 0.025, 72)
    df = pd.DataFrame({"Date": dates, "ret": s})
    p = tmp_path / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def test_ladder_runs_with_baselines_only(tmp_path):
    ar_path = _make_asset_returns(tmp_path)
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--asset-returns", str(ar_path),
        "--oos-start", "2016-01-31",
        "--out", str(out),
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "ladder_returns.csv").exists()
    assert (out / "ladder_metrics.csv").exists()
    metrics = pd.read_csv(out / "ladder_metrics.csv", index_col=0)
    # equal_weight and risk_parity must always be present.
    assert "equal_weight" in metrics.index
    assert "risk_parity" in metrics.index


def test_ladder_runs_with_full_model(tmp_path):
    ar_path = _make_asset_returns(tmp_path)
    full_path = _make_model_returns(tmp_path, seed=5, name="full")
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--asset-returns", str(ar_path),
        "--oos-start", "2016-01-31",
        "--full-model-returns", str(full_path),
        "--out", str(out),
        "--n-tournament-trials", "500",
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    metrics = pd.read_csv(out / "ladder_metrics.csv", index_col=0)
    assert "full_model" in metrics.index
    # DSR column present and in [0, 1].
    dsr = metrics.loc["full_model", "dsr_vs_tournament"]
    assert 0.0 <= float(dsr) <= 1.0
