# Insignia Testnet Deployment & Emulator Guide

This guide reflects the latest orchestration findings from 2026-04-16 and the repository updates applied from the target-achieving third run.

## Highlights

- Sentinel status is `SECURE_AND_IMPROVING`, with Phase 5 conditions satisfied and the breach-rate target achieved.
- Commit-reveal defaults are enabled in config and tracked with timestamp metrics.
- Default wallet layout now covers 1 owner, 1 validator, and 12 miners.
- Market diversification now includes BTC, ETH, SOL, AVAX, and ADA.
- Monitoring assets exist in both `monitoring/` and `testnet/docker-compose.testnet.yml`.
- PC-VH-006 (symbol diversity enforcement) is now deployed; remaining work is empirical confirmation of the projected Sybil reduction in live conditions.

## Quick local stack

```bash
cd subnet
docker compose -f testnet/docker-compose.testnet.yml up -d
bash testnet/scripts/check_chain_connectivity.sh
bash testnet/scripts/check_wallet_balances.sh
```

## Commit-reveal validation targets

| Metric | Target |
|---|---:|
| effectiveness | >= 0.76 |
| critical floor | >= 0.667 |
| validator latency severity | < 0.05 |
| prediction timing severity | < 0.03 |
| consecutive validations | >= 6 |

## Current defaults

### Commit-reveal

- enabled: true
- commit_window_seconds: 30
- reveal_window_seconds: 15
- nonce_bits: 128
- max_reveal_attempts: 3
- late_reveal_penalty: 1.0
- grace_period_seconds: 2.0

### Validation timing

- min_prediction_lead_time_seconds: 35
- validator_latency_penalty_weight: 0.28
- high_latency_threshold_ms: 1800
- commit_rate_threshold: 0.75
- commitment_violation_weight: 0.012
- expected_commit_reveal_effectiveness: 0.76
- required_validation_streak: 6
- selective_reveal_warning_streak: 1
- selective_reveal_penalty_streak: 2
- selective_reveal_zero_streak: 3

### Ensemble detection

- correlation_threshold: 0.80
- entropy_threshold_lower: 0.20
- symbol_diversity_threshold: 0.33
- fusion_strategy: bayesian_model_averaging
- response_vote_threshold: 3

### Stable MCP model routing

- enabled: configurable
- stable_per_run: true
- assignment_seed: configurable
- route_names / route_ids: provided by the MCP routing server
- each simulated miner/trader agent receives a fixed `assigned_route` for the full run
- optional `assigned_model_profile` metadata can be attached for reproducible diagnostics
- assignments are emitted in simulation/emulator results so route diversity is auditable
- this is used to model decentralized intelligence diversity during tuning without re-sampling route quality every epoch

### Symbol diversity enforcement (PC-VH-006)

- enabled: true
- minimum_trading_pairs: 3
- max_symbol_dominance: 0.60
- warning dominance ratio: 1.35
- critical dominance ratio: 2.0
- base penalty: 0.10
- escalation factor: 1.5
- maximum penalty: 0.50
- grace period generations: 2

### Consensus integrity

- weight_entropy_minimum: 1.45
- cross_validator_score_variance_max: 0.18
- validator_rotation_max_consecutive_epochs: 4
- validator_agreement_threshold: 0.17
- collusion_detection_lookback_epochs: 12

## Monitoring endpoints

| Service | URL |
|---|---|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Emulator metrics | http://localhost:8001/metrics |

## Key report-aligned metrics

- `insignia_commit_timestamp`
- `insignia_reveal_timestamp`
- `insignia_no_reveal_streak`
- `insignia_timing_attack_composite_severity`
- `insignia_trading_pair_activity`
- `insignia_ensemble_signal`
- `insignia_btc_eth_dominance_ratio`
- `insignia_symbol_diversity_enforcement`
- `insignia_commit_reveal_effectiveness`

## Emulator topology

- validator nodes / wallets: 1
- miner nodes / wallets: 12
- simulated benchmark population: 14 agents
  - L1: 6 honest + 4 adversarial
  - L2: 3 honest + 1 adversarial

## Note on parameter counts

The orchestration brief references a 75-parameter optimization headline. The repository still exposes a broader parameter surface in code because the 10-metric L2 scorer and additional safeguards remain enabled for realism.
