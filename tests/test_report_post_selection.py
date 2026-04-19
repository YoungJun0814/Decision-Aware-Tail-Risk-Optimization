"""Smoke test for the post-selection reporting script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_post_selection.py"


def _write_port(tmp_path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-07-31", periods=96, freq="ME")
    r = rng.normal(0.008, 0.03, 96)
    p = tmp_path / "port.csv"
    pd.DataFrame({"Date": dates, "ret": r}).to_csv(p, index=False)
    return p


def _write_candidates(tmp_path: Path, seed: int = 1, K: int = 30) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-07-31", periods=96, freq="ME")
    X = rng.normal(0.003, 0.03, (96, K))
    df = pd.DataFrame(X, index=dates, columns=[f"cand_{k}" for k in range(K)])
    df.index.name = "Date"
    p = tmp_path / "candidates.csv"
    df.to_csv(p)
    return p


def test_report_runs_port_only(tmp_path):
    port = _write_port(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPT),
           "--port-returns", str(port),
           "--n-trials", "500",
           "--threshold", "-0.10",
           "--n-bootstrap", "300",
           "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads((out / "report.json").read_text())
    assert data["n_trials_for_dsr"] == 500
    assert 0.0 <= data["dsr"]["value"] <= 1.0
    assert "reality_check" not in data  # no candidates provided


def test_report_runs_with_candidates(tmp_path):
    port = _write_port(tmp_path)
    cand = _write_candidates(tmp_path)
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPT),
           "--port-returns", str(port),
           "--candidates", str(cand),
           "--threshold", "-0.10",
           "--n-bootstrap", "300",
           "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads((out / "report.json").read_text())
    assert data["n_trials_for_dsr"] == 30  # K overrides --n-trials
    assert "reality_check" in data
    assert "tournament_bootstrap" in data
    assert 0.0 <= data["reality_check"]["p_value"] <= 1.0
    assert (out / "report.md").exists()
