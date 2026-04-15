# Sentinel Attack Surveillance

## Current status

- Phase: 4 - Attack Surveillance
- System posture: `SECURE_AND_IMPROVING`
- Transition: Phase 5 transition viable
- Active sentinel metrics: breach rate `0.0005`, honest score `0.94`, score separation `0.758`
- Commit-reveal effectiveness: `0.700` (passes the `0.667` floor with 6 consecutive validations)
- Convergence detected: false
- Reset triggers: `SOFT=false`, `HARD=false`, `FULL=false`
- Persistent warning: Sybil pressure from BTCUSDT:ETHUSDT imbalance

## Alert levels

| Level | Trigger | Action |
|---|---|---|
| INFO | breach severity <= 0.05 | continue monitoring |
| WARNING | breach severity > 0.05 | bias search toward breach reduction |
| CRITICAL | severity > 0.15 or 3 consecutive increases | trigger HARD reset workflow |
| EMERGENCY | 5+ simultaneous breaches | trigger FULL reset workflow |

## Monitoring configuration

- Frequency: 60 seconds
- Moving average window: 5 generations
- Convergence detection: 3 consecutive severity increases
- Commit/reveal telemetry: commit timestamp, reveal timestamp, no-reveal streak
- Timing composite metric: `insignia_timing_attack_composite_severity`

## Attack catalog

The current Phase 5 gate evaluates **19 post-commit-reveal vectors**. The
repository still keeps richer detector telemetry for extensions like
`sybil_collusion_graph`, `temporal_attack_pattern`, and
`cross_layer_correlation`, but the sentinel transition decision is based on the
post-CR 19-vector posture.

### Legacy vectors retained in code

1. overfitting_exploitation
2. model_plagiarism
3. single_metric_gaming
4. sybil_attack
5. copy_trading
6. random_baseline_discrimination
7. adversarial_dominance
8. insufficient_separation
9. score_concentration
10. validator_latency_exploitation
11. prediction_timing_manipulation
12. miner_validator_collusion
13. weight_entropy_violation
14. cross_validator_score_variance
15. validator_rotation_circumvention
16. validator_agreement_anomaly
17. collusion_temporal_pattern
18. weight_manipulation
19. cross_layer_attack

### Rich telemetry extensions retained in the repository

20. selective_revelation
21. statistical_anomaly
22. behavioral_anomaly
23. temporal_attack_pattern
24. sybil_collusion_graph
25. cross_layer_correlation

## Current notable severities

| Vector | Severity | Tier | Notes |
|---|---:|---|---|
| V3 Sybil Attack | 0.268 | warning | structural gap; commit-reveal is not meant to solve Sybil pressure |
| V8 Commitment Violation / Front-Running | 0.0402 | warning but below breach line | trending downward under sustained commit-reveal enforcement |
| Temporal attack pattern | 0.01 | info | down ~98% from the earlier 0.51 persistent-risk benchmark |
| Selective revelation | 0.004 | info | sharply reduced by post-CR enforcement |
| Wash trading | 0.003 | info | effectively neutralized in the current sentinel posture |
| Pump / dump | 0.003 | info | effectively neutralized in the current sentinel posture |
| Remaining post-CR vectors | < 0.05 | info | 17 of 19 vectors now sit in the info band |

## Program risk mapping

The sentinel mapped all 8 `program.md` persistent risks to current post-CR
severities. Seven are already at `INFO`; the only remaining structural gap is
Sybil pressure, which should be addressed by deploying `PC-VH-006` (Symbol
Diversity Enforcement).

## Reset protocols

### SOFT

- Double the scalarized breach-rate weight
- Inject 5 random individuals
- Preserve all elites
- Refocus researcher on the active attack family

### HARD

- Rebuild population with 30% Pareto elites
- Add 30% researcher best configurations
- Replace 40% with random candidates
- Reset convergence counters and restart optimizer generation numbering

### FULL

- Save state and tear down the local stack
- Recreate subnet and infrastructure
- Restart from fresh population and cleared researcher history

## Known issues

- The dominant BTCUSDT:ETHUSDT ratio remains the main warning-level weakness.
- `PC-VH-006` (Symbol Diversity Enforcement) is the recommended next mitigation to close 19/19 vector coverage.
- The earlier harsh simulation benchmark (`0.124` breach rate, `0.847` honest score) should now be treated as calibration context, not the current system-level security posture.
- Historical docs may use older vector numbering or the richer telemetry catalog; repository docs should prefer the post-CR sentinel framing when discussing transition readiness.
