# V14-R1 Empirical Separation Validation

**Generated:** 2026-07-04T02:23:20.737409+00:00
**Config source:** encode_defaults (Phase 5 proxy; V14-R1-CORRECTED-KP lives in orchestrator MongoDB)
**Parameters:** trials=3, epochs=5, trading_steps=120

## §9 Separation Gate (threshold >= 0.9)

| Metric | Value |
|---|---|
| separation_mean | 0.9004 |
| separation_min | 0.9004 |
| separation_max | 0.9004 |
| separation_stdev | 0.0000 |
| honest_mean_avg | 0.9007 |
| adversarial_mean_avg | 0.0004 |
| breach_rate_mean | 0.00 |
| **gate_passed** | **True** |
| n_trials_passing_gate | 3/3 |

## Per-Trial Results

| Trial | Separation | Honest | Adversarial | Breach Rate | Elapsed |
|---|---|---|---|---|---|
| 0 | 0.9004 | 0.9007 | 0.0004 | 0.00 | 46.8s |
| 1 | 0.9004 | 0.9007 | 0.0004 | 0.00 | 42.5s |
| 2 | 0.9004 | 0.9007 | 0.0004 | 0.00 | 41.8s |

## Verdict

✅ **GATE PASSED**, all 3 trials clear §9 separation >= 0.9.
The merged anti-gaming fix (PR #34) + signal-driven SybilMiner penalty hold
empirically. The cycle may proceed to step 3 (sentinel re-evaluation).

_JSON report: `v14_r1_empirical_validation_2026-07-04T02-21-09.json`_