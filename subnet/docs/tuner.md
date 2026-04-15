# Tuner

## NSGA-II configuration

- Population: 30
- Generations: 20
- Objectives: 4
- Crossover: SBX, probability 0.9, eta 15
- Mutation: polynomial mutation, eta 20
- Seeds: EXP-116, EXP-118

## Objectives

1. maximize honest score
2. minimize breach rate
3. minimize score variance
4. maximize score separation

The implementation stores these as minimization targets:

- `neg_honest_score`
- `breach_rate`
- `score_variance`
- `neg_separation`

## Pareto maintenance

- Keep the full Pareto front for analysis
- Select the knee point by normalized distance to the ideal point
- Export Pareto size and diversity to Prometheus

## Current frontier snapshot

- Configured search regime: 20 generations x 30 population
- Optimization target: breach_rate < `5e-6`, honest_score > `0.97`, separation > `0.75`
- Best seed-to-date result remains EXP-140 with breach_rate `2.5e-05`
- Remaining optimization gap to target breach_rate: roughly 5x
- Best operating region remains non-converged, supporting continued Phase 5 tuning

## Report-aligned defaults

### L1 weights

- directional accuracy: 0.22
- sharpe ratio: 0.18
- max drawdown: 0.14
- stability: 0.16
- overfitting penalty: 0.14
- feature efficiency: 0.06
- latency: 0.10

### L2 weights in code

The report summarizes a 6-metric headline allocation, but the repository keeps a 10-metric L2 scorer for risk controls. The checked-in defaults were updated to the nearest compatible mapping:

- realized_pnl: 0.21
- omega: 0.15
- max_drawdown: 0.12
- win_rate: 0.07
- consistency: 0.17
- model_attribution: 0.08
- execution_quality: 0.05
- annualized_volatility: 0.05
- sharpe_ratio: 0.05
- sortino_ratio: 0.05

## Report-driven tuning guidance

- The simulation layer is intentionally harsher than the autoresearch loop baseline: 100 epochs, 14 agents, 5 pairs, and elevated adversarial pressure.
- Commit-reveal remains a validated prerequisite, with observed effectiveness `0.723` clearing the `0.667` floor.
- Highest-priority objectives should bias toward the two dominant surveillance risks:
  - `sybil_collusion_graph = 0.63`
  - `temporal_attack_pattern = 0.51`
- Economic-mechanism knobs are the strongest current leverage:
  - identity bonding
  - stake-based consensus
  - commit-reveal enforcement
- EXP-140 (decentralized identity verification with bonding) is the leading seed reference, with EXP-141 (Bayesian model averaging) the best ensemble-style backup.
