# Changelog

## 2026-04-14 - Orchestration report synchronization

- upgraded the tuning stack toward AttackDetector v8 behavior with commitment violation scoring, selective revelation escalation, and expanded anomaly/correlation checks
- updated default weights and validation timing hyperparameters from the latest orchestration findings
- added `composite_integrity_scorer.py` for EXP-023 style integrity scoring
- expanded protocol and testnet config with SOL, AVAX, and ADA market diversification support
- enriched simulator telemetry for commit/reveal timestamps, no-reveal streaks, validator timing, ensemble signals, and convergence tracking
- expanded Prometheus metrics with commit/reveal timestamps, timing attack composite, trading-pair activity, and ensemble signals
- updated autoresearch logging for experiment ids, radical levels, and report-aligned TSV output
- documented Phase 4 status, Phase 5 transition viability, NSGA-II defaults, and researcher/sentinel workflows
