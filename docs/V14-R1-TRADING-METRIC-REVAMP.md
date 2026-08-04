# V14-R1-CORRECTED-KP Trading Metric Revamp (2026-08-03)

## Amendment Reference

**AMEND-2026-08-03-trading-metric-revamp**

This document records the rebase of the V14-R1-CORRECTED-KP configuration's
trading weight space, aligning it with the 2026-08-03 trading metric revamp
amendment.

## Background

The v3 orchestrator pre-flight check `config_matches_current_parameter_space`
failed because the V14-R1-CORRECTED-KP trading section still carried the
pre-2026-08-03 composite (9-key weight space including `trading_realized_pnl`
and `trading_win_rate`), while the current parameter space expects the
8-key `annualized_return_v2` scoring schema.

## Rebase Specification

### Keys Dropped

| Key                     | Reason                                              |
|------------------------|------------------------------------------------------|
| `trading_realized_pnl`  | Replaced by `trading_annualized_return` as the profitability headline metric |
| `trading_win_rate`      | Demoted to reported-only diagnostic (no longer a scored trading weight) |

### Key Added

| Key                        | Role                                         |
|---------------------------|----------------------------------------------|
| `trading_annualized_return` | Profitability headline metric (replaces `trading_realized_pnl`) |

### Post-Rebase Trading Weight Space (8 keys, equal weights)

| Key                        | Weight |
|---------------------------|--------|
| `trading_annualized_return` | 0.125  |
| `trading_sharpe_ratio`      | 0.125  |
| `trading_sortino_ratio`     | 0.125  |
| `trading_max_drawdown`      | 0.125  |
| `trading_calmar_ratio`      | 0.125  |
| `trading_profit_factor`    | 0.125  |
| `trading_fill_quality`     | 0.125  |
| `trading_execution_cost`   | 0.125  |
| **Sum**                     | **1.0** |

### Renormalization

- **Factor:** x1/0.94 (the combined mass of the two dropped keys was 0.06)
- **Method:** Equal weights (0.125 × 8 = 1.0)
- **Source rationale:** No authoritative pre-revamp 9-weight vector was found
  in agent_memory (`tuner_state`, `sentinel_state`, `simulator_state`) or
  MongoDB `experiment_configs` (procedure=`insignia_subnet_tuner`, config_id
  regex `V14-R1|V13-R3`). Equal weights were used as the fallback per task
  instructions.

## Scoring Schema

- **Schema:** `annualized_return_v2`
- **Pre-revamp weight count:** 9
- **Post-revamp weight count:** 8
- **Sum check:** 1.0 (exact)

## Constraints (Do-NOT List)

1. Do NOT decode the quarantined warm-start artifact
   `results/tuner_v14_r1_warmstart_seed_2026-07-04T02-39-07.json` with the new
   parameter space.
2. Do NOT write `criterion_met=true` or any promotion status.
3. Do NOT push a `parameter_space.py` promotion diff or run `btcli apply`
   (no stale parameter-space promotion).
4. Do NOT use `procedure=insignia_subnet_tuner` for V14-R1 verification writes.

## Reference Config File

The rebased configuration is persisted as:
`subnet/reference_configs/knee_point_V14-R1-CORRECTED-KP.json`

## Parent Configuration

- **Parent:** `V13-R3-KP-020-a3c7`
- **Correction rationale:** Fix 4× separation overestimation by strengthening
  anti-gaming, overfitting, promotion, and identity parameters based on
  empirical validation findings.

## Grounded Root Cause

V13-R3 failed empirical validation because `SimulationHarness.run` in
`subnet/tuning/simulation.py` (scoring inline around lines 860–930) applies
`_scaled(multiplier=0.10)` only to `CopycatMiner` and `CopyTrader`.
`SybilMiner`, `OverfittingMiner`, `SingleMetricGamer`, and `PartnerGamer`
have no penalty path. `SybilMiner` scores higher than honest (0.9163 vs
0.9151).
