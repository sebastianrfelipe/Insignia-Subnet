# Tuner

## NSGA-II configuration

- Population: 30
- Generations: 20
- Objectives: 4
- Crossover: SBX, probability 0.9, eta 15
- Mutation: polynomial mutation, eta 20
- Active seed lineage: EXP-116, EXP-118, EXP-140, EXP-141
- NSGA-II profile: v13 R2
- Surrogate model: Gaussian Process, trained from 93 empirical data points
- Surrogate fit: R2 = 0.93
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
- Optimization target achieved: breach_rate < `5e-6`, honest_score > `0.97`, separation > `0.75`
- Winning knee point: `V13-R2-KP-020-a7f2`
- Knee-point metrics:
  - breach_rate: `3.5e-6`
  - honest_score: `0.9795`
  - separation: `0.953`
  - variance: `0.0009`
- Target was first achieved by generation 12; generations 13-20 refined the Pareto front
- Final Pareto front size: 21
- Hypervolume improved from `0.0018` to `0.0161` (+794%)
- Current system-level sentinel posture remains `SECURE_AND_IMPROVING`; optimization now shifts from target-finding to empirical validation of the predicted optimum

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

- The third orchestration run confirmed the surrogate prediction: a stacked structural defense can beat the `5e-6` target rather than merely approach it.
- Commit-reveal remains a validated prerequisite, with system-level effectiveness `0.76` and simulator-estimated pre/post effectiveness `0.801`.
- The active knee-point defense stack is:
  - decentralized identity verification (`identity_bond = 0.08`)
  - stake-based consensus
  - Bayesian model averaging
  - commit-reveal
- `PC-VH-006` is now deployed and should be treated as part of the active tuning baseline rather than a future recommendation.
- Remaining work is empirical:
  - verify the surrogate-predicted `3.5e-6` breach rate in live simulation
  - measure realized Sybil reduction from `0.195` toward the projected `0.08`
