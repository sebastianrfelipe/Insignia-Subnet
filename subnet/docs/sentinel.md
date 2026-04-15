# Sentinel Attack Surveillance

## Current status

- Phase: 4 - Attack Surveillance
- Transition: Phase 5 transition viable
- Latest simulation benchmark: 100 epochs, 14 agents, 5 trading pairs
- Headline simulation metrics: breach rate `0.124`, honest score `0.847`
- Commit-reveal effectiveness: `0.723` (passes the `0.667` floor)
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

The simulator and detector now track the legacy 19-vector catalog plus the report-driven expansion used for ensemble surveillance.

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

### Report-driven surveillance extensions

20. selective_revelation
21. statistical_anomaly
22. behavioral_anomaly
23. temporal_attack_pattern
24. sybil_collusion_graph
25. cross_layer_correlation

## Current notable severities

| Vector | Severity | Tier | Notes |
|---|---:|---|---|
| sybil_collusion_graph | 0.63 | high | primary unresolved risk; tied to cluster coordination and pair imbalance |
| temporal_attack_pattern | 0.51 | high | second-highest risk in the orchestration run |
| selective_revelation | 0.45 | moderate | contained by commit/reveal streak penalties but still visible |
| wash_trading | 0.42 | moderate | still part of the middle attack band |
| cross_layer_correlation | 0.39 | moderate | elevated when timing and ensemble votes align |
| mev_extraction | 0.35 | moderate | remains a live execution-layer concern |
| statistical_anomaly | 0.32 | moderate | now treated as part of the monitored middle band |
| behavioral_anomaly | 0.28 | low-moderate | tracked, but not a top-tier issue in this run |
| validator_latency_exploitation | < 0.05 target band | controlled | commit/reveal remains validated |
| prediction_timing_manipulation | < 0.05 target band | controlled | co-benefit from commit/reveal remains intact |

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
- The simulation layer is materially harsher than the autoresearch loop; do not compare the `0.124` simulation breach rate directly to the `2.5e-05` experiment optimum without noting the environment mismatch.
- Historical docs may use older vector numbering; repository docs should prefer the detector naming above.
