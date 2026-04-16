# Sentinel Attack Surveillance

## Current status

- Phase: 5 - Target achieved / empirical validation
- System posture: `SECURE_AND_IMPROVING`
- Transition: Phase 5 conditions met
- Active sentinel metrics: breach rate `0.0005`, honest score `0.94`, score separation `0.758`
- Commit-reveal effectiveness: `0.76` (passes the `0.667` floor with a 14% margin)
- Composite integrity score: `0.978`
- Convergence detected: false
- Reset triggers: `SOFT=false`, `HARD=false`, `FULL=false`
- Persistent warning: Sybil pressure from BTCUSDT:ETHUSDT imbalance is declining and should be further reduced by PC-VH-006

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
| V3 Sybil Attack | 0.195 | warning | declining slowly; PC-VH-006 should accelerate the reduction toward the `0.08` target |
| V8 Commitment Violation / Front-Running | 0.019 | info | materially improved from `0.0402` |
| Core 9-vector base surface | < 0.05 | info | all core vectors are now well below threshold |
| Temporal attack pattern | no anomaly detected | info | no new temporal attack signatures detected in the third run |
| New attack patterns | none detected | info | sentinel/tuner coordination reported no new anomaly clusters |

## Program risk mapping

The third orchestration run confirms that the remaining structural gap is now
primarily an empirical validation problem rather than a design-gap problem:

- the optimizer target has been hit at `3.5e-6`
- PC-VH-006 is deployed
- Sybil severity is down from `0.268` to `0.195`
- the next step is to confirm the projected production reduction toward `0.08`

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

- The dominant BTCUSDT:ETHUSDT ratio remains the main warning-level weakness, though it is now on a declining trajectory.
- `PC-VH-006` is deployed, but its projected 60-70% Sybil reduction still needs live empirical confirmation.
- The earlier harsh simulation benchmark (`0.124` breach rate, `0.847` honest score) should now be treated as calibration context, not the current system-level security posture.
- Historical docs may use older vector numbering or the richer telemetry catalog; repository docs should prefer the post-CR sentinel framing when discussing transition readiness.
