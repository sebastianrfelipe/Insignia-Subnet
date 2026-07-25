# GBDT Autoresearch — Cycle 34 Run State Summary (TSLA)

> **Generated**: 2026-07-25 | **Procedure**: gbdt_autoresearch | **Symbol**: TSLA | **Cycle**: 34  
> **Status**: cycle_34_complete (catastrophic_regression) | **Escalation Level**: 3  
> **Convergence State**: catastrophic_regression | **Stagnation Counter**: 4  
> **Last Improvement Cycle**: 30 (HV=0.0931)

---

## 1. Cycle 34 Experiment Results

### AUTORES-GBDT-082 — F1-Only Warm-Start

| Field | Value |
|---|---|
| **Status** | `completed` |
| **Experiment Type** | `f1_only_warm_start` |
| **Hypothesis** | Dropping Sharpe entirely and focusing on F1 with wider fwd windows will recover F1 performance |
| **Parent** | AUTORES-GBDT-017 (cycle 12 champion, F1=0.585) |
| **Best F1** | 0.1803 |
| **N Pareto** | 1 |
| **Sharpe** | N/A (F1-only mode) |
| **Hypervolume** | 0.005 |
| **Save Dir** | `/mnt/Synth/Analysis/TSLA/auto_ml_results/runs/AUTORES-GBDT-082` |
| **Duration** | 5,778 sec (~96 min) |
| **Started** | 2026-07-25T12:03:37Z |
| **Completed** | 2026-07-25T13:39:55Z |
| **NSGA Config** | pop_size=30, n_gen=40 |
| **Fwd Windows** | bullish_fwd=[80,288], bearish_fwd=[80,288] |
| **UP_TH** | bullish=[0.005,0.07], bearish=[0.005,0.07] |
| **Features Dropped** | 33 (legacy pruned set) |
| **Env: SHARPE_AGGREGATION** | none |
| **Env: NSGA_OBJECTIVES** | f1_only |
| **Env: GEN_GAP_CONSTRAINT** | 0.4 |

**Verdict**: F1-only optimization failed to recover F1. Wider fwd windows [80,288] diluted the label signal, producing only 1 Pareto solution with F1=0.180 — 24% regression from cycle 33 best (0.237).

---

### AUTORES-GBDT-083 — Asymmetric Regime Thresholds

| Field | Value |
|---|---|
| **Status** | `failed_non_convergence` |
| **Experiment Type** | `asymmetric_regime_thresholds` |
| **Hypothesis** | Asymmetric regime thresholds will break the Sharpe anti-correlation by creating imbalanced bullish/bearish label distributions |
| **Parent** | AUTORES-GBDT-081 |
| **Best F1** | 0.076 |
| **N Pareto** | 0 |
| **Killed At** | Gen 11 |
| **Failure Reason** | F1=0.06 (below 0.10 floor), 0% constraint feasibility, best_f1=0.076 is only 32% of previous champion |
| **NSGA Config** | pop_size=40, n_gen=50 |
| **Fwd Windows** | bullish_fwd=[12,60], bearish_fwd=[120,320] |
| **UP_TH** | bullish=[0.005,0.03], bearish=[0.01,0.2] |
| **Env: BULLISH_SHARPE_MIN** | 0.1 |
| **Env: GEN_GAP_CONSTRAINT** | 0.5 |
| **Env: SHARPE_AGGREGATION** | arithmetic_mean |
| **Features Dropped** | 33 (legacy pruned set) |

**Verdict**: BULLISH_SHARPE_MIN=0.1 constraint created an empty feasible region — no GBDT config on this dataset can achieve bullish Sharpe > 0.1. Asymmetric thresholds made Sharpe anti-correlation worse by creating extreme label imbalance.

---

### AUTORES-GBDT-084 — Top-10 Feature Ablation

| Field | Value |
|---|---|
| **Status** | `failed_non_convergence` |
| **Experiment Type** | `top_features_ablation` |
| **Hypothesis** | Reducing to top-10 features may eliminate noise features that contribute to Sharpe anti-correlation |
| **Parent** | AUTORES-GBDT-017 |
| **Best F1** | 0.08 |
| **N Pareto** | 0 |
| **Killed At** | Gen 16 |
| **Failure Reason** | F1=0.08 (below 0.10 floor), 0% constraint feasibility, HV declining. Same degenerate pattern as 083 |
| **NSGA Config** | pop_size=30, n_gen=40 |
| **Fwd Windows** | bullish_fwd=[12,60], bearish_fwd=[120,320] |
| **UP_TH** | bullish=[0.005,0.07], bearish=[0.005,0.07] |
| **Env: BULLISH_SHARPE_MIN** | 0.1 |
| **Env: GEN_GAP_CONSTRAINT** | 0.5 |
| **Env: SHARPE_AGGREGATION** | arithmetic_mean |
| **Feature Subset** | top_10_from_cycle_12_champion |

**Verdict**: Top-10 feature ablation destroyed model capacity. Only 10 features insufficient for GBDT to learn meaningful regime patterns. BULLISH_SHARPE_MIN=0.1 still too strict for this feature set.

---

## 2. Pareto Front Summary

| Experiment | Status | N Pareto Solutions | Best F1 | Pareto Front CSV | all_results.csv |
|---|---|---|---|---|---|
| AUTORES-GBDT-082 | completed | **1** | 0.1803 | `/mnt/Synth/Analysis/TSLA/auto_ml_results/runs/AUTORES-GBDT-082/pareto_front.csv` | `/mnt/Synth/Analysis/TSLA/auto_ml_results/runs/AUTORES-GBDT-082/all_results.csv` |
| AUTORES-GBDT-083 | killed gen 11 | **0** | 0.076 | N/A (killed before completion) | N/A |
| AUTORES-GBDT-084 | killed gen 16 | **0** | 0.08 | N/A (killed before completion) | N/A |

**Total Pareto solutions across cycle 34: 1** (worst in experiment history)

> **Note**: Pareto front CSV and all_results.csv filesystem existence could not be verified remotely. Only AUTORES-GBDT-082 (completed) would have these files; 083/084 were killed before completion.

---

## 3. Hypervolume Estimates

| Cycle | Hypervolume | Δ vs Prior | Best F1 | Convergence State |
|---|---|---|---|---|
| 28 | 0.0280 | -0.0182 | 0.2441 | stagnating |
| 29 | 0.0480 | +0.0200 | 0.2704 | stagnating |
| 30 | **0.0931** | +0.0451 | 0.2937 | partial_recovery |
| 31 | *(no convergence_metrics doc)* | — | 0.593† | — |
| 32 | 0.0480 | -0.0451 | 0.2752 | stagnation_sharpe_blocked |
| 33 | 0.0250 | -0.0230 | 0.2367 | stagnation_sharpe_blocked |
| **34** | **0.0050** | **-0.0200** | **0.1803** | **catastrophic_regression** |

† Cycle 31 best F1=0.593 (075) but Sharpe=0 invalidated the result; convergence_metrics document was not persisted separately.

**Hypervolume trend**: Steady decline from peak 0.0931 (cycle 30) → 0.005 (cycle 34). Four consecutive cycles of regression.

---

## 4. Convergence State

| Property | Value |
|---|---|
| **Convergence State** | `catastrophic_regression` |
| **Stagnation Counter** | 4 (cycles 31-34 with no improvement) |
| **Last Improvement Cycle** | 30 (HV=0.0931) |
| **Early Stopping Patience** | 15 generations |
| **Hypervolume Improvement Threshold** | 0.001 |
| **Stagnation Window Size** | 5 cycles |
| **Avg Fitness** | 0.112 |
| **Gen Gap Mean** | 0.55 |
| **Feasibility Rate** | 0 (no feasible Sharpe-constrained solutions) |
| **Best HV History** | [0.0462, 0.028, 0.048, 0.0931, 0.048, 0.025, 0.005] |

---

## 5. Researcher State

| Property | Value |
|---|---|
| **_id** | `gbdt_autoresearch` |
| **Status** | `cycle_33_complete` → `cycle_34_complete` |
| **Last Cycle** | 33 → 34 |
| **Active Symbol** | TSLA |
| **Escalation Level** | 3 |
| **Dataset** | features_v5_TSLA_dollar_20260718_064110_options.csv |
| **Dataset Rows** | 100,242 (options variant) |
| **Dataset Hash** | `01ed8a13642f75faa131f55dac0e576af26f5b50ef2667627aee36cc34cbb543` |
| **Feature Version** | v5 |
| **Bar Mode** | dollar bars, 1min |
| **Options Features** | enabled (18 options features) |
| **Champion** | AUTORES-GBDT-081 (F1=0.2367) → AUTORES-GBDT-082 (F1=0.1803) |
| **Overall Champion** | GBDT-TSLA-C1-006 (F1=0.6388) |
| **Queued Experiments** | AUTORES-GBDT-082, 083, 084 (now completed) |
| **Next Cycle Experiments** | AUTORES-GBDT-085, 086, 087 |
| **Sharpe Zero Diagnosis** | `arithmetic_mean_symmetric_opposite_signs` |

---

## 6. Historical Trend (Cycles 28-34)

### F1 Trend

```
Cycle 28: 0.2441 ████████████▌
Cycle 29: 0.2704 █████████████▍
Cycle 30: 0.2937 ██████████████▋
Cycle 31: 0.5930 ██████████████████████████████ (075, Sharpe=0 invalidated)
Cycle 32: 0.2752 █████████████▋
Cycle 33: 0.2367 ████████████
Cycle 34: 0.1803 █████████
```

### Hypervolume Trend

```
Cycle 28: 0.0280 ███
Cycle 29: 0.0480 █████
Cycle 30: 0.0931 █████████▎     ← PEAK
Cycle 31: (not recorded)
Cycle 32: 0.0480 █████
Cycle 33: 0.0250 ██▌
Cycle 34: 0.0050 █▌             ← FLOOR
```

### Pareto Front Size Trend

```
Cycle 28: 27 solutions
Cycle 29: 30 solutions
Cycle 30:  8 solutions (071, high quality)
Cycle 31: 40 solutions (073, wide but shallow)
Cycle 32: 40 solutions (degenerate, all non-dominated)
Cycle 33: 21 solutions (081)
Cycle 34:  1 solution  (082 only) ← WORST
```

---

## 7. Key Diagnostics

### Dead Zones (Unreachable Regions)
- **F1 > 0.20**: Unreachable in cycle 34
- **Sharpe > 0**: Unreachable (structural anti-correlation)
- **F1 > 0.55**: Unreachable since cycle 12 (overall champion GBDT-TSLA-C1-006 = 0.639)
- **Constraint feasibility**: Zero with BULLISH_SHARPE_MIN=0.1

### Root Cause Analysis (Cycle 34)
1. **BULLISH_SHARPE_MIN=0.1 constraint** (083/084): Eliminated all feasible configs — no GBDT config on this dataset achieves bullish Sharpe > 0.1
2. **Wider fwd windows [80,288]** (082): Diluted label signal, producing degenerate F1=0.18
3. **Top-10 feature ablation** (084): Destroyed model capacity — 10 features insufficient for GBDT regime classification
4. **Sharpe anti-correlation** (structural): bullish=+0.092, bearish=-0.092 are symmetric opposites; arithmetic mean = 0

### Sharpe Anti-Correlation History
| Cycle | Bullish Sharpe | Bearish Sharpe | Aggregation | Result |
|---|---|---|---|---|
| 28-30 | +0.092 | -0.092 | harmonic | 0 (undefined for mixed signs) |
| 31 | +0.092 | -0.092 | harmonic | 0 (floored) |
| 32 | +0.092 | -0.092 | arithmetic | 0 (symmetric cancellation) |
| 33 | +0.092 | -0.092 | decoupled | F1 regressed (0.137-0.237) |
| 34 | +0.092 | -0.092 | none/arithmetic | F1 regressed further (0.076-0.180) |

---

## 8. Escalation Status

| Property | Value |
|---|---|
| **Current Level** | 3 (change mutation strategy) |
| **Stagnation Cycles** | 4 (cycles 31-34) |
| **Level 4 Trigger** | stagnation_cycles >= 12 (reset population, random restart) |
| **Level 5 Trigger** | stagnation_cycles >= 15 (full system restart) |
| **Decision** | HOLD at level 3 with strategic pivot |
| **Rationale** | Constraint configuration is primary bottleneck, not search space |

### Escalation History
| Cycle | Level | Action | Reason |
|---|---|---|---|
| 31 | 3 | hold | Sharpe=0 is structural (harmonic-mean), needs code-level fix |
| 32 | 3 | hold | Arithmetic-mean fix deployed but insufficient — symmetric opposite signs |
| 33 | 3 | hold | All decoupled Sharpe approaches regressed F1 |
| 34 | 3 | hold | Constraint config is primary bottleneck; strategic pivot needed |

---

## 9. Cycle 35 Strategy (Next Cycle)

Per researcher_insights and audit_log:

| Experiment ID | Strategy | Description |
|---|---|---|
| **AUTORES-GBDT-085** | Remove ALL Sharpe constraints | F1-only optimization with champion seed from cycle 12 (AUTORES-GBDT-017) |
| **AUTORES-GBDT-086** | Epsilon-constraint Sharpe > -0.1 | F1+Sharpe joint objectives, narrow UP_TH, 10-feature restoration |
| **AUTORES-GBDT-087** | Population restart with elite injection | 80% random + 20% elite injection from cycle 30 champion (071) |

**Key changes needed**:
1. Remove BULLISH_SHARPE_MIN=0.1 constraint entirely
2. Use narrow fwd windows [12,60]/[12,60] or symmetric [40,200] instead of [80,288]
3. Restore minimum 10-feature set (opt_gex_0dte, opt_net_premium, opt_gex_weekly, ofi_5, opt_call_volume, opt_put_volume, ret_10, ret_30, vol_120, macd_hist)
4. Consider population restart to escape degenerate NSGA-II convergence

---

## 10. Data Source Inventory

| Source | Collection | Query Filter | Records Found |
|---|---|---|---|
| experiment_configs | MongoDB | `{procedure: "gbdt_autoresearch", experiment_id: {$in: ["082","083","084"]}}` | 3 |
| researcher_state | MongoDB | `{_id: "gbdt_autoresearch"}` | 1 |
| convergence_metrics | MongoDB | `{procedure: "gbdt_autoresearch", symbol: "TSLA"}` cycles 28-34 | 6 (incl. 2 dupes for cycle 34) |
| researcher_insights | MongoDB | `{procedure: "gbdt_autoresearch", symbol: "TSLA"}` cycles 31-34 | 8 |
| audit_log | MongoDB | `{procedure: "gbdt_autoresearch", symbol: "TSLA", cycle: 34}` | 6 |
| pareto_front.csv | Filesystem | `/mnt/Synth/Analysis/TSLA/auto_ml_results/runs/AUTORES-GBDT-082/` | Not verified remotely |
| all_results.csv | Filesystem | Same as above | Not verified remotely |

---

*End of Cycle 34 Run State Summary*