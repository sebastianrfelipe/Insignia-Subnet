# Tuner

## NSGA-II configuration

- Population: 30
- Generations: 20
- Objectives: 4
- Crossover: SBX, probability 0.9, eta 15
- Mutation: polynomial mutation, eta 20
- Active seed lineage: EXP-116, EXP-118, EXP-140, EXP-141
- NSGA-II profile: v13
- Surrogate model: enabled, trained from 93 empirical data points
- Active surrogate variable count: 41
- Sampling: Latin Hypercube + elite seeding
- Elite seeds injected: EXP-140, EXP-141, EXP-134, EXP-132, EXP-133, EXP-135

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
- Current system-level sentinel posture is already `SECURE_AND_IMPROVING`; the optimizer is now focused on closing the residual numeric gap rather than rescuing a failing defense surface

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

- realized_pnl: 0.18
- omega: 0.12
- max_drawdown: 0.12
- win_rate: 0.05
- consistency: 0.18
- model_attribution: 0.11
- execution_quality: 0.09
- annualized_volatility: 0.05
- sharpe_ratio: 0.05
- sortino_ratio: 0.05

## Report-driven tuning guidance

- The harsh 100-epoch / 14-agent / 5-pair simulation remains useful as calibration, but the second orchestration run elevated the sentinel system view to the primary operating reference.
- Commit-reveal remains a validated prerequisite, with observed effectiveness `0.700` clearing the `0.667` floor for 6 consecutive validations.
- The active warning vectors are now much narrower:
  - `V3 Sybil Attack = 0.268`
  - `V8 Commitment Violation / Front-Running = 0.0402`
- Economic-mechanism knobs remain the strongest current leverage:
  - identity bonding
  - stake-based consensus
  - commit-reveal enforcement
  - Bayesian model averaging
- EXP-140 (decentralized identity verification with bonding) remains the leading seed reference, with EXP-141 (Bayesian model averaging) the strongest ensemble backup.
- Deploy `PC-VH-006` (Symbol Diversity Enforcement) to close the last structural Sybil gap while NSGA-II searches the residual 5x breach-rate gap.
