# Cycle 34 Step 6 — Final Persistence Summary

**Procedure:** `gbdt_autoresearch`  
**Symbol:** TSLA  
**Cycle:** 34  
**Timestamp:** 2026-07-25T17:30:00Z  
**Inserted By:** orchestrator-cycle34-step6

---

## MongoDB Documents Persisted

### 1. convergence_metrics (1 document)
- **Collection:** `convergence_metrics`
- **ID:** `6a64f4fa4c9f3f50f46f8bb6`
- **Key Fields:**
  - `convergence_state`: `catastrophic_regression`
  - `hypervolume`: 0.005
  - `stagnation_counter`: 4
  - `pareto_front_size`: 1
  - `last_improvement_cycle`: 30
  - `best_hypervolume_history`: [0.0462, 0.028, 0.048, 0.0931, 0.048, 0.025, 0.005]

### 2. researcher_insights (1 document)
- **Collection:** `researcher_insights`
- **ID:** `6a64f4fa4c9f3f50f46f8bb8`
- **Key Findings:**
  - BULLISH_SHARPE_MIN=0.1 constraint eliminated all feasible configs
  - Wider fwd windows [80,288] diluted label signal
  - Top-10 feature ablation destroyed model capacity
  - Sharpe anti-correlation is structural
  - **Next strategy:** Remove all Sharpe constraints, champion seed, epsilon-constraint, population restart

### 3. experiment_configs (3 documents for cycle 35)
- **AUTORES-GBDT-085** (`6a64f4fa4c9f3f50f46f8bba`): Constraint-free F1-only with champion seed from AUTORES-GBDT-017
- **AUTORES-GBDT-086** (`6a64f4fa4c9f3f50f46f8bbc`): Epsilon-constraint Sharpe > -0.1 with F1+Sharpe joint objectives
- **AUTORES-GBDT-087** (`6a64f4fa4c9f3f50f46f8bbe`): Population restart with 80% random + 20% elite injection from AUTORES-GBDT-071

### 4. researcher_state (updated)
- **Collection:** `researcher_state`
- **Filter:** `{_id: "gbdt_autoresearch"}`
- **Updates Applied:**
  - `status` → `cycle_34_complete`
  - `last_cycle` → 34
  - `cycle` → 34
  - `last_hypervolume` → 0.005
  - `last_best_f1` → 0.1803
  - `champion_experiment_id` → `AUTORES-GBDT-082`
  - `convergence_detection.convergence_state` → `catastrophic_regression`
  - `convergence_detection.stagnation_count` → 4
  - `auto_cycler.current_cycle` → 34
  - `auto_cycler.escalation.current_level` → 3
  - `auto_cycler.escalation.consecutive_no_improvement` → 4
  - Escalation history appended with cycle 34 entry
  - `queued_experiments` → ["AUTORES-GBDT-085", "AUTORES-GBDT-086", "AUTORES-GBDT-087"]

### 5. audit_log (2 documents)
- **cycle_34_complete** (`6a64f4fa4c9f3f50f46f8bc0`): Full cycle 34 completion audit with diagnostics
- **step6_cycle_034_completion** (`6a64f4fa4c9f3f50f46f8bc2`): Step 6 persistence confirmation

---

## Cycle 34 Diagnostics Summary

| Metric | Value |
|--------|-------|
| Best F1 | 0.1803 (AUTORES-GBDT-082) |
| Hypervolume | 0.005 |
| Pareto Front Size | 1 |
| Feasibility Rate | 0% |
| Stagnation Counter | 4 |
| Escalation Level | 3 |
| Convergence State | catastrophic_regression |
| HV Trend | Declining from 0.0931 (cycle 30) → 0.005 (cycle 34) |
| F1 Trend | Catastrophic regression (24% drop from cycle 33) |

## Root Cause
1. BULLISH_SHARPE_MIN=0.1 constraint created empty feasible region
2. Wider fwd windows [80,288] diluted label signal
3. Top-10 feature ablation destroyed model capacity

## Pending (Requires Deployer)
- Cycle 034 JSON filesystem snapshot
- TSV row append to `results/tuner_experiments.tsv`