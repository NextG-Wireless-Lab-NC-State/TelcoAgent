# Reproducing the Paper Experiments

This guide is the canonical entry point for **reviewers** and **co-authors**
who want to recreate the TSFM forecasting baseline numbers, plots, and
ablations reported in the paper.

> **Pre-condition.** A single CUDA-capable GPU (≥ 24 GB) and roughly
> ~20 GB of free disk for the full sweep grid. CPU-only is supported but
> ~30× slower.

---

## 1. Environments (two conda envs needed)

The repo uses **two** conda envs because Chronos-2 requires
`chronos-forecasting==2.0.0` and that version conflicts with the newer
`transformers` we need for MOMENT / uni2ts / Toto.

```bash
# Main env — Moirai-family, MOMENT, Toto, Mamba4Cast, TelcoAgent runtime
conda env create -f envs/environment.yml
conda activate telcoagent

# Slim sibling env — Chronos-2 only
conda env create -f envs/environment-chronos2.yml
conda activate telco-chronos-test
```

Once both envs exist you do **not** activate them manually for the sweep
runners — each runner is invoked via `conda run -n <env>`.

---

## 2. Determinism

Every sweep runner accepts `--seed` (default `42`) and the harness pins
all RNGs (Python `random`, NumPy, PyTorch CPU/CUDA, `cudnn.deterministic`,
`torch.use_deterministic_algorithms(True, warn_only=True)`) before the
first forward pass — see [`scripts/baselines/foundation_utils.py:set_seed`](../scripts/baselines/foundation_utils.py).

The seed and the **installed wheel version** of the inference library
are recorded into every per-station JSON under
`extra_info.seed` and `extra_info.lib_version`. Reviewers can audit the
exact wheel that produced each number without consulting `environment.yml`.

---

## 3. Paper baseline re-run (3 main + 4 supplementary Moirai variants)

Run sequentially — total wall time ~6–9 h on one RTX 4090.

```bash
# Or use the bundled script that does all 7 in order:
bash scripts/baselines/_rerun_all_paper_baselines.sh
```

Equivalent per-model commands:

| Model | Command | Env |
|---|---|---|
| **Chronos-2** (paper main) | `conda run -n telco-chronos-test python scripts/baselines/run_chronos2_context_sweep.py --seed 42` | `telco-chronos-test` |
| **MOMENT** (paper main) | `conda run -n telcoagent python scripts/baselines/run_moment_context_sweep.py --seed 42` | `telcoagent` |
| **Moirai** (paper main, picked by best MASE) | `conda run -n telcoagent python scripts/baselines/run_moirai_family_context_sweep.py --model Salesforce/moirai-moe-1.0-R-base --seed 42` | `telcoagent` |
| Moirai 1.1 small / base / large | replace `--model` with `Salesforce/moirai-1.1-R-{small,base,large}` | `telcoagent` |
| Moirai-MoE 1.0 small | replace `--model` with `Salesforce/moirai-moe-1.0-R-small` | `telcoagent` |

Each command sweeps 81 context lengths × 115 stations and writes per-station
JSON + CSV under `output/7d_prediction_tsfm/<safe_id>_h7d_ctx_sweep/`.

Two variants intentionally not in the paper main table:

| Model | Why supplementary |
|---|---|
| **Toto** (`Datadog/Toto-Open-Base-1.0`) | Best raw numbers but out-of-scope for the paper's "cross-channel TSFM" story; kept for the appendix |
| **Mamba4Cast** | Channel-independent SSM baseline; requires `git clone https://github.com/automl/Mamba4Cast` at the repo root + `mamba-ssm==2.2.2 causal-conv1d==1.4.0`. Not bundled because of the prebuilt-wheel CUDA-version dance. |

---

## 4. Naive seasonal baseline (always emitted)

For every `(model, ctx, station)` the harness also computes the
seasonal-naive(s = 24 h) forecast. This shows up in two places:

- per-station JSON: `extra_info.naive_seasonal_metrics`
- per-context aggregate: `<ctx_dir>/naive_summary.json`

Use this as the paper's reference row — by construction the naive
forecast's MASE is bounded around 1.0 on this dataset (Hyndman seasonal
denominator), so any model with `MASE < 1` is beating the trivial
24 h-shift baseline.

---

## 5. Metrics

Two scale-free metrics are reported per KPI and as cross-KPI means:

- **MASE** (`metrics.per_kpi.<KPI>.MASE`): Hyndman & Koehler (2006) seasonal
  MASE with `s = 24`; denominator is the seasonal-naive MAE on the **input
  window** (81 days), so the value is horizon- and ctx-independent.
- **nRMSE** (`metrics.per_kpi.<KPI>.nRMSE`): RMSE divided by the test-target
  range `max(truth) − min(truth)`; in our setup the test target is fixed at
  the last 168 h, so nRMSE is comparable across ctx values.

Raw `RMSE`, `MAE`, and `sMAPE` are kept per KPI for supplementary tables.

---

## 6. Paper figures and tables

| Artifact | Source dir | Generation script | Output PDF/PNG |
|---|---|---|---|
| Fig. ctx-sweep (3 baselines) | `output/7d_prediction_tsfm/{chronos2_ctx_sweep_v2, moirai-moe-1.0-R-base_ctx_sweep, moment_ctx_sweep}/` | `scripts/plots/ctx_sweep_h7d.py` (TBD — refactor of `/tmp/plot_ctx_sweep_combined.py`) | `paper/figures/fig_ctx_sweep.pdf` |
| Fig. horizon-sweep | (separate horizon-sweep runs; see §7) | `scripts/plots/horizon_sweep.py` | `paper/figures/fig_horizon_sweep.pdf` |
| Tab. main results | same dirs as ctx-sweep, ctx fixed at each model's best | `scripts/plots/main_table.py` (TBD) | `paper/figures/tab_main_results.tex` |

Each plot script reads `sweep_summary.csv` from the relevant model dir
and writes into `paper/figures/`. Naming convention:
`fig<N>_<short_slug>.{pdf,png}` for figures, `tab<N>_<slug>.tex` for tables.

---

## 7. Multi-horizon experiments (paper supplementary)

For the horizon-scaling supplementary plot (forecast 7 d → 70 d), run
each baseline once per horizon with `--horizon-days <H>`. Note that as
horizon grows the maximum input window shrinks (`88 − H` days), so
`--context-days` must be ≤ that bound. Example:

```bash
for H in 7 14 21 28 35 42 49 56 63 70; do
  C=$((88 - H))
  conda run -n telcoagent python scripts/baselines/run_moirai_family_context_sweep.py \
    --model Salesforce/moirai-moe-1.0-R-base --seed 42 \
    --horizon-days $H --context-days $C
done
```

---

## 8. Verification (smoke + drift)

```bash
# 1-station smoke for any runner (~10–60 s)
conda run -n telcoagent python scripts/baselines/run_toto_context_sweep.py \
  --stations station_A_10 --context-days 28 --seed 42 --output-dir /tmp/toto_smoke

# Drift check vs the legacy archive (must be ≤ 0.05 MASE per KPI)
conda run -n telcoagent python scripts/baselines/diff_one_station.py \
  --new output/7d_prediction_tsfm/chronos2_ctx_sweep_v2/ctx_672h/station_A_10.json \
  --old output/7d_prediction_tsfm/_archive_pre_seed/chronos2_ctx_sweep_v2/ctx_672h/station_A_10.json \
  --metric MASE --tol 0.05
```

The harness also runs `assert_nonflat_prediction(predictor)` once before
every sweep starts; if the model degenerates to flat output (we hit this
historically with `chronos-forecasting==2.2.2`) the sweep aborts before
wasting hours.

---

## 9. Where everything lives

| Location | Contents |
|---|---|
| `output/7d_prediction_tsfm/<safe_id>_ctx_sweep/` | Paper-grade sweep results (post `--seed 42` re-run) |
| `output/7d_prediction_tsfm/_archive_pre_seed/` | Legacy results (pre-seed, kept for diffing) |
| `output/28d_only_tsfm/chronos2/` | Reference 28 d Chronos-2 run (paper supplementary methods) |
| `output/plots/` | Older plot assets (preserved; new ones land in `paper/figures/`) |
| `paper/figures/` | Final paper figures (PDF + PNG) |
| `output/<model>/naive_summary.json` (per ctx) | Seasonal-naive baseline aggregate per (model, ctx) |

---

## 10. Caveats

- **`Mamba4Cast/`** must be `git clone`d into the repo root and the
  pretrained checkpoint placed at `models/mamba4cast_2l_1024_conv_i5e5.pth`
  for the supplementary SSM baseline; see the runner docstring at
  [`scripts/baselines/run_mamba4cast.py`](../scripts/baselines/run_mamba4cast.py).
- **MOMENT** has a fixed input length of 512 h (= 21 d). Contexts longer
  than that are truncated to the trailing 512 h inside the model — the
  CSV header still records the requested ctx for traceability.
- **Moirai-2.x** is rejected by `run_moirai_family_context_sweep.py`
  because the v2 release dropped multivariate Any-Variate Attention. Use
  Moirai-1.1 or Moirai-MoE for cross-channel results.
- **Determinism caveat**: `torch.use_deterministic_algorithms(True,
  warn_only=True)`. A handful of uni2ts / xFormers kernels still take a
  non-deterministic CUDA path and emit a one-line warning. Outputs match
  to within 1e-5 across reruns; we accept this as paper-grade.

---

## 11. Provenance

This file was generated as part of the paper-grade upgrade plan
at commit `f14560a`. The 7-baseline re-run launched on **2026-05-02** under
`--seed 42`. Old (pre-seed) outputs are preserved under
`output/7d_prediction_tsfm/_archive_pre_seed/` for diffing.
