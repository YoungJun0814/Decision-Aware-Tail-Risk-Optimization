# Repository Status — Canonical Truth for Existence of Artifacts

**Commit reference:** this document reflects the state of `origin/main` at commit `ffe6b59` and is updated as remediation commits land. If any other doc contradicts this file, **this file is authoritative**.

Generated 2026-04-19 in response to P0.4 (docs↔code drift) of the [Remediation Plan](REMEDIATION_PLAN.md).

---

## TL;DR

Several older docs (`README.md`, `docs/THESIS_CURRENT_MODEL_PIPELINE.md`, `docs/COMPLETE_MODEL_DOCUMENTATION.md`, `docs/IMPROVEMENT_STRATEGY.md`, `TRIPLE_TARGET_RESULTS.md`) were written against a larger local workspace that included tournament result directories and Phase 18 overlay scripts that were never pushed to this repository. The pushed code alone cannot reproduce the canonical Sharpe 1.0724 / MDD -9.44% / Bootstrap 57.98% numbers because key scripts and fitted-regime CSVs are missing.

Do not treat those older docs as reproducible specifications until the missing artifacts are restored or the docs are rewritten.

---

## Scripts

| Referenced Path | Exists in repo? | Notes |
|---|---|---|
| `src/gen_regime_4state.py` | ✅ | Fits HMM on **full 2007–2025 sample** → has look-ahead bias. |
| `src/gen_regime_4state_pit.py` | ❌ | **Does not exist.** P0.1 of the remediation plan implements it. |
| `scripts/run_correlation_guard_v7_exclusion.py` | ✅ | Dispatcher loads the two scripts below via `importlib.util.spec_from_file_location`. |
| `scripts/run_correlation_guard_v8_tiered.py` | ✅ | |
| `scripts/run_phase18_paper_safe_ablation.py` | ❌ | Imported by the Guard v7 dispatcher. Every end-to-end run of the overlay **will fail at import**. |
| `scripts/run_phase18_nonleveraged_v2_benchmark.py` | ❌ | Same as above. |
| `scripts/run_phase18_nonleveraged_v2_structural_tournament.py` | ❌ | Referenced only in docs. |
| `scripts/run_triple_precision_search.py` | ✅ | Upstream search only; does not include overlay. |
| `scripts/run_triple_precision_round2.py` | ✅ | |
| `scripts/run_triple_precision_round3.py` | ✅ | |
| `scripts/verify_triple_champion.py` | ✅ | |
| `run_walkforward.py` | ✅ | `--pit_hmm` flag is currently a no-op (file-existence check only). |

## Data / Regime CSVs

| Referenced Path | Exists? | Notes |
|---|---|---|
| `data/processed/regime_4state.csv` | ✅ | Produced by `gen_regime_4state.py`; full-sample HMM (look-ahead). |
| `data/processed/regime_4state_pit.csv` | ❌ | **Missing.** `data_loader.get_regime_4state()` silently falls back to the look-ahead CSV. Every downstream claim of "PIT-safe" is therefore unsupported by the pushed code. |
| `data/processed/prob_data.csv` | ✅ | |
| `data/processed/regime_comparison.csv` | ✅ | |

## Results Directories

| Referenced Path | Exists? | Notes |
|---|---|---|
| `results_runpod/walkforward/` | ✅ | |
| `results_runpod/correlation_guard_v7/` | ✅ | |
| `results_runpod/correlation_guard_v8/` | ✅ | |
| `results_runpod/phase17/` | ✅ | |
| `results_runpod/nvidia_tournament_round3_group_constraints/` | ❌ | Canonical winner referenced in `README.md:16-17` and `docs/THESIS_CURRENT_MODEL_PIPELINE.md:19-21`. Not pushed. |
| `results_runpod/nvidia_tournament_round4_mdd_reopt/` | ❌ | Round 4 verification dir. Not pushed. |
| `results_runpod/phase18_triple_precision_round3/` | ❌ | "Pre-NVIDIA triple champion" baseline. Not pushed. |
| `results_runpod/phase18_triple_verification/` | ❌ | Verification report. Not pushed. |

## Canonical Metrics — Reproducibility Status

| Metric | Reported Value | Reproducible from pushed code alone? |
|---|---:|---|
| Sharpe | 1.0724 | ❌ (requires missing overlay scripts + tournament dirs) |
| Annual return | 10.1035% | ❌ |
| Max drawdown | -9.4414% | ❌ |
| Bootstrap P(MDD ≥ -10%) | 57.98% | ❌ |

Upstream-only artifacts (walk-forward results in `results_runpod/walkforward/`) are reproducible.

---

## Active Remediation

See [REMEDIATION_PLAN.md](REMEDIATION_PLAN.md). P0 in flight:

- [x] **P0.2** — Encoder argument plumbing fixed (`aa36bd2`).
- [x] **P0.3** — `batch_tail_mean` vs distributional CVaR disambiguated (`3635252`).
- [ ] **P0.4** — This document is step 1; step 2 is either pushing the missing scripts from the author's local workspace or rewriting the dependent docs.
- [ ] **P0.1** — Implement real PIT-HMM (`gen_regime_4state_pit.py` + per-fold refit) and rerun headline experiments.

## What a Reader Should Do

1. Treat `README.md` headline metrics as **historical**, not current.
2. For any claim of "PIT-safe regime", verify `data/processed/regime_4state_pit.csv` exists first.
3. For any claim that cites `nvidia_tournament_round3/4` or `phase18_triple_*`, treat as **unreproducible from this repo alone**.
4. The code that *is* in this repo — upstream BL+CVaR, walk-forward harness, regime head, loss functions — is reproducible and is what the thesis methodology chapter describes.
