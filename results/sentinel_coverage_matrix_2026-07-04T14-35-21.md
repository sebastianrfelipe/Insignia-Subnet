# Sentinel Coverage Matrix, 19 Post-CR Surveillance Vectors

**Generated:** 2026-07-04T14:36:08.766434+00:00
**Config source:** encode_defaults (Phase 5 proxy; V14-R1-CORRECTED-KP lives in orchestrator MongoDB)
**Parameters:** epochs=5, trading_steps=120

## Summary

| Metric | Value |
|---|---|
| Post-CR vectors evaluated | 19 |
| Post-CR vectors breached | 2 |
| Post-CR breach rate | 0.1053 |
| Detector total vectors (incl. rich telemetry) | 28 |
| Detector total breached | 2 |
| Detector breach rate | 0.0714 |
| Detector mean severity | 0.0774 |
| Detector max severity | 0.9041 |

## Coverage breakdown

| Coverage state | Count |
|---|---|
| ✅ CLOSED (signal-driven, empirically verified) | 2 |
| 🟡 STATIC-FLOOR (multiplier backstop, pending signal pipeline) | 5 |
| ⚙️ CONFIG (hyperparameter defense) | 9 |
| 📊 AGGREGATE (combined effect) | 1 |

## Per-vector matrix

| # | Vector | Breached | Severity | Tier | Coverage | Adversary type(s) | Defense path |
|---|---|---|---|---|---|---|---|
| 1 | `overfitting_exploitation` | ✅ no | 0.0 | low | 🟡 STATIC-FLOOR | OverfittingMiner | _OVERFITTER_MULTIPLIER (static floor 0.0001) |
| 2 | `model_plagiarism` | ✅ no | 0.0 | low | 🟡 STATIC-FLOOR | CopycatMiner | _COPYCAT_MULTIPLIER (static floor 0.0001) |
| 3 | `single_metric_gaming` | ✅ no | 0.0 | low | 🟡 STATIC-FLOOR | SingleMetricGamer | _SINGLE_METRIC_MULTIPLIER (static floor 0.0001) |
| 4 | `sybil_attack` | ✅ no | 0.0 | low | ✅ CLOSED | SybilMiner | signal-driven: sybil_pressure × detection_sensitivity × correlation_penalty |
| 5 | `copy_trading` | ✅ no | 0.0 | low | 🟡 STATIC-FLOOR | CopyTrader | _COPYTRADER_MULTIPLIER (static floor 0.0001) |
| 6 | `random_baseline_discrimination` | ❌ YES | 0.9041 | high | 📊 AGGREGATE | RandomMiner (noise baseline, NOT adversarial per §5.1) | scoring discrimination (no penalty path, this vector checks the scorer separates signal from noise) |
| 7 | `adversarial_dominance` | ✅ no | 0.0 | low | 📊 AGGREGATE | any adversary | all penalty paths combined (no adversary scores above honest) |
| 8 | `insufficient_separation` | ✅ no | 0.0 | low | ✅ CLOSED | all adversaries (aggregate) | §9 separation gate (>= 0.90), all penalty paths combined |
| 9 | `score_concentration` | ✅ no | 0.0716 | low | ⚙️ CONFIG | aggregate (HHI of miner_scores) | emission reverse-sigmoid + pairing marginal-contribution credit |
| 10 | `validator_latency_exploitation` | ✅ no | 0.0351 | low | ⚙️ CONFIG | validator timing exploit | validation_timing config (min_prediction_lead_time, validator_latency_penalty_weight) |
| 11 | `prediction_timing_manipulation` | ✅ no | 0.025 | low | ⚙️ CONFIG | miner timing exploit | validation_timing config (min_prediction_lead_time, commitment_violation_weight) |
| 12 | `miner_validator_collusion` | ✅ no | 0.0 | low | 🟡 STATIC-FLOOR | ColludingResearcher, colluder_trader | _COLLUDER_MULTIPLIER (0.0001) + 0.40 non-transferability + consensus_integrity config |
| 13 | `weight_entropy_violation` | ✅ no | 0.0 | low | ⚙️ CONFIG | validator weight manipulation | consensus_integrity config (weight_entropy_minimum 1.45) |
| 14 | `cross_validator_score_variance` | ✅ no | 0.0 | low | ⚙️ CONFIG | validator disagreement exploit | consensus_integrity config (cross_validator_score_variance_max 0.18) |
| 15 | `validator_rotation_circumvention` | ✅ no | 0.0 | low | ⚙️ CONFIG | validator rotation exploit | consensus_integrity config (validator_rotation_max_consecutive_epochs 4) |
| 16 | `validator_agreement_anomaly` | ✅ no | 0.0 | low | ⚙️ CONFIG | validator agreement exploit | consensus_integrity config (validator_agreement_threshold 0.17) |
| 17 | `collusion_temporal_pattern` | ❌ YES | 0.7429 | high | 🟡 STATIC-FLOOR | ColludingResearcher (temporal) | _COLLUDER_MULTIPLIER (0.0001) + collusion_detection_lookback_epochs (12) |
| 18 | `weight_manipulation` | ✅ no | 0.0 | low | ⚙️ CONFIG | L1/L2 weight skew | pairing config (marginal_contribution_weight 0.50, fixed_pair_correlation_threshold 0.85) |
| 19 | `cross_layer_attack` | ✅ no | 0.1667 | low | ⚙️ CONFIG | cross-layer timing sync | validation_timing + pairing config (combined) |

## Pending signal-driven work (STATIC-FLOOR vectors)

- **`overfitting_exploitation`**, EXP-ADVERSARY-COVERAGE-002 §2, replace with IS/OOS gap signal
- **`model_plagiarism`**, fingerprint_correlation_threshold config exists; multiplier is backstop
- **`single_metric_gaming`**, EXP-ADVERSARY-COVERAGE-002 §3, replace with metric concentration + entropy
- **`copy_trading`**, copy_trade_correlation_threshold config exists; multiplier is backstop
- **`miner_validator_collusion`**, collusion_detection_lookback_epochs config; multiplier is backstop

## Breach annotations (non-penalty-path breaches)

### `random_baseline_discrimination` (severity 0.9041, high)

**Detector description:** Random avg=0.8144 vs honest avg=0.9007 (ratio=0.90)

**Annotation:** NOT an adversary leak. RandomMiner is the §5.1 noise baseline (not adversarial); the synthetic scorer gives it ~0.81 by design (vs honest ~0.92, ratio 0.89 > 0.60 threshold). This vector checks scoring discrimination, not adversary suppression. Closing it would require changing the synthetic score generation, not the penalty paths. Acceptable for the V14-R1 gate, RandomMiner is already excluded from the adversarial set in test_simulation_separation.py.

### `collusion_temporal_pattern` (severity 0.7429, high)

**Detector description:** Max temporal corr=0.820, flagged pairs=10, lookback=12

**Annotation:** NOT an adversary leak. The harness hardcodes miner_validator_temporal_corr = 0.82 for sybil+validator_0 pairs (simulation.py:1094) as a synthetic test signal to exercise the detector. The _COLLUDER_MULTIPLIER (0.0001) zeroes the colluder's SCORE (miner_validator_collusion vector passes at severity 0.0), but this vector checks the temporal-correlation SIGNAL, which is a separate synthetic telemetry field the score penalty does not touch. To close this vector, the harness would need to reduce the synthetic correlation when the colluder is penalized, or the detector threshold (0.7) would need to be raised. Neither is an anti-gaming fix.

## Verdict

✅ **All adversary-type vectors defended**, zero adversary leaks across the 19-vector surface.
   - Adversary-type vectors (overfitting, plagiarism, single_metric, sybil, copy_trading,
     miner_validator_collusion, adversarial_dominance, insufficient_separation): all severity 0.0.
   - Non-penalty-path breaches (2): ['random_baseline_discrimination', 'collusion_temporal_pattern']
     These are synthetic harness signals / config-tuning gaps, NOT adversary leaks
     (see breach annotations above). The anti-gaming fix is working correctly.

The merged anti-gaming fix (PR #34) + signal-driven SybilMiner penalty (PR pending)
close the sentinel adversary surface empirically. The cycle may proceed to step 4
(tuner NSGA-II fold) once the sentinel agent confirms this matrix against the live
V14-R1-CORRECTED-KP config from MongoDB.

_JSON report: `sentinel_coverage_matrix_2026-07-04T14-35-21.json`_