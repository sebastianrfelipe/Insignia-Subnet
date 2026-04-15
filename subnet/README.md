# Insignia Subnet

**A decentralized network where independent participants compete to build the best predictive models - and prove they work in the real world.**

Built on [Bittensor](https://bittensor.com) for the Sovereign Infrastructure Hackathon (March 2026).

**Status:** Phase 4 (Attack Surveillance) - Phase 5 transition viable  
**System posture:** `SECURE_AND_IMPROVING`  
**Best reported checkpoint:** EXP-140 with breach_rate `0.000025`, honest_score `0.9748`

---

## What Is This?

Insignia is a two-layer competitive network for producing high-quality ML models and validating that they actually work when deployed.

- **Layer 1 (Model Competition):** miners train predictive models and are scored across 7 weighted metrics.
- **Layer 2 (Deployment Validation):** promoted models are wrapped into strategies and scored with the repository's 10-metric L2 risk stack.
- **Cross-Layer Feedback:** Layer 2 outcomes feed back into Layer 1 rankings.
- **Commit-Reveal Validation:** the timing defense path now holds at composite effectiveness `0.700`, above the `0.667` acceptance floor for 6 consecutive validations.

---

## Current orchestration snapshot

- 6-agent architecture: deployer, simulator, sentinel, tuner, researcher, coder
- 8 implemented agent archetypes in simulation: Honest, Overfitter, Copycat, SingleMetricGamer, Sybil, Random, HonestTrader, CopyTrader
- 14-agent benchmark simulation mix: 9 honest, 5 adversarial across L1/L2
- Active sentinel posture: breach_rate `0.0005`, honest_score `0.94`, score_separation `0.758`
- No convergence detected and no reset triggers fired (`SOFT`, `HARD`, `FULL` all false)
- 75-parameter orchestration headline, with the repository retaining a broader 10-metric L2 implementation and expanded parameter space in code
- 41-variable NSGA-II v13 surrogate profile runs on top of the repository's broader 64-parameter surface
- Current optimization spec: 20 generations, population 30, 4 objectives, 93 empirical surrogate points
- Open optimization gap: best autoresearch result `2.5e-05` vs target `5e-06`
- Persistent warning: Sybil pressure driven by BTCUSDT:ETHUSDT imbalance
- Historical hard-environment simulation headline remains useful context: breach_rate `0.124`, honest_score `0.847`

## L1 weights

| Metric | Weight |
|---|---:|
| directional_accuracy / penalized_f1 | 0.22 |
| sharpe_ratio / penalized_sharpe | 0.18 |
| max_drawdown | 0.14 |
| stability / variance_score | 0.16 |
| overfitting_penalty | 0.14 |
| feature_efficiency | 0.06 |
| latency | 0.10 |

## L2 weights used in repository defaults

The orchestration report summarizes a 6-metric headline split, but the codebase intentionally keeps a 10-metric L2 scorer for additional risk controls. The updated compatible defaults are:

| Metric | Weight |
|---|---:|
| realized_pnl | 0.18 |
| omega_ratio | 0.12 |
| max_drawdown | 0.12 |
| win_rate | 0.05 |
| consistency | 0.18 |
| model_attribution | 0.11 |
| execution_quality | 0.09 |
| annualized_volatility | 0.05 |
| sharpe_ratio | 0.05 |
| sortino_ratio | 0.05 |

---

## Attack surveillance

### Active post-commit-reveal security gate

The current sentinel operating model treats **19 post-commit-reveal vectors** as
the Phase 5 security gate. The repository still keeps richer telemetry for
additional extensions, but the active transition decision is based on the
19-vector post-CR posture.

### Legacy 19-vector catalog retained in code

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

- selective_revelation
- statistical_anomaly
- behavioral_anomaly
- temporal_attack_pattern
- sybil_collusion_graph
- cross_layer_correlation

### Current warning vectors from the second orchestration run

- `V3 Sybil Attack`: `0.268` (structural warning, not solved by commit-reveal alone)
- `V8 Commitment Violation / Front-Running`: `0.0402` (below the 0.05 breach line and trending down)
- 17 of 19 post-CR vectors are already in the `INFO` band
- Recommended next deployment: `PC-VH-006` (Symbol Diversity Enforcement)

See `docs/sentinel.md` for the full alert and reset workflow.

---

## Data diversification

The repository now treats these as first-class trading pairs for simulation and testnet configuration:

- BTCUSDT / BTC-USDT-PERP
- ETHUSDT / ETH-USDT-PERP
- SOLUSDT / SOL-USDT-PERP
- AVAXUSDT / AVAX-USDT-PERP
- ADAUSDT / ADA-USDT-PERP

This diversification is the primary mitigation for the persistent Sybil warning
tied to BTC/ETH dominance. The latest sentinel report specifically recommends
deploying `PC-VH-006` (Symbol Diversity Enforcement) to close 19/19 coverage.

---

## Autoresearch highlights

- 25 experiments executed from baseline `EXP-116`
- 17 kept, 8 discarded
- Best result: `EXP-140` (decentralized identity verification with bonding)
- Runner-up: `EXP-141` (Bayesian Model Averaging)
- Best-performing family: architecture redesign (7 experiments, 5 kept)
- Weakest family: temporal pattern analysis (3 experiments, 1 kept)
- Higher radical levels (3-4) produced the largest gains, but also the most discards
- Next queued tranche: `EXP-142` through `EXP-166`, focused on economic security
  mechanisms such as dynamic bonding, stake slashing, quadratic staking,
  commit-reveal v2, VDF-based commit-reveal, and reputation bonding

---

## Monitoring

Prometheus and Grafana assets live in `monitoring/` and `testnet/docker-compose.testnet.yml`.

Notable report-aligned metrics now include:

- `insignia_commit_timestamp`
- `insignia_reveal_timestamp`
- `insignia_no_reveal_streak`
- `insignia_timing_attack_composite_severity`
- `insignia_trading_pair_activity`
- `insignia_ensemble_signal`

---

## Project structure

```text
subnet/
├── insignia/
├── neurons/
├── tuning/
│   ├── attack_detector.py
│   ├── autoresearch_loop.py
│   ├── composite_integrity_scorer.py
│   ├── optimizer.py
│   └── simulation.py
├── testnet/
│   ├── config.py
│   ├── emulator.py
│   └── scripts/
├── monitoring/
├── docs/
└── results/
```

---

## Quick start

```bash
cd subnet
uv sync
python3 -m tuning.simulation
python3 -m tuning.attack_detector
python3 -m tuning.autoresearch_loop --max-experiments 5
python3 -m tuning.orchestrator --mode optimize --generations 20 --population 30
```

Monitoring stack:

```bash
cd subnet
docker compose -f monitoring/docker-compose.yml up -d
```

Testnet health scripts:

```bash
bash testnet/scripts/check_chain_connectivity.sh
bash testnet/scripts/check_wallet_balances.sh
```
