# Tuner

## NSGA-II configuration

- Population: 30
- Generations: 20
- Objectives: 4
- Crossover: SBX, probability 0.9, eta 15
- Mutation: polynomial mutation, eta 20
- Active seed lineage: EXP-116, EXP-118, EXP-140, EXP-141
- NSGA-II profile: v13 R3
- Surrogate model: Gaussian Process, trained from 93 empirical data points
- Surrogate fit: R2 = 0.96
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
- Winning knee point: `V13-R3-KP-020-a3c7` (supersedes `V13-R2-KP-020-a7f2`)
- Knee-point metrics:
  - breach_rate: `2.6e-6`
  - honest_score: `0.9808`
  - separation: `0.963`
  - variance: `0.00081`
- Knee point stable since generation 7 (13 consecutive generations as knee), angle score `0.94`
- Improvement vs R2 knee: breach_rate 25.7% better, honest_score 0.13% better, variance 10% better, separation 1.05% better
- Final Pareto front size: 26
- Hypervolume improved to `0.0189` (vs `0.0161` at R2)
- Full reference: [../reference_configs/knee_point_V13-R3.json](../reference_configs/knee_point_V13-R3.json)
- Current system-level sentinel posture remains `SECURE_AND_IMPROVING`; optimization now shifts from target-finding to empirical validation of the predicted optimum

## Report-aligned defaults

### Model weights (researcher)

- directional accuracy: 0.22
- sharpe ratio: 0.18
- max drawdown: 0.14
- stability: 0.16
- overfitting penalty: 0.14
- feature efficiency: 0.06
- latency: 0.10

### Trading weights (trader) in code

The report summarizes a 6-metric headline allocation, but the repository keeps a 9-metric trading scorer for risk controls. The `model_attribution` metric was removed when pairing became genetic-algorithm-assigned (a miner cannot influence which model it is paired with), and its weight was redistributed. The checked-in defaults are:

- realized_pnl: 0.20
- omega: 0.13
- max_drawdown: 0.14
- win_rate: 0.06
- consistency: 0.20
- execution_quality: 0.10
- annualized_volatility: 0.05
- sharpe_ratio: 0.06
- sortino_ratio: 0.06

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
  - verify the surrogate-predicted `2.6e-6` breach rate in live (online) simulation
  - measure realized Sybil reduction from `0.195` toward the projected `0.08`
