# Tuner

## NSGA-II configuration

- Population: 30
- Generations: 50
- Objectives: 4
- Crossover: SBX, probability 0.9, eta 15
- Mutation: polynomial mutation, eta 20
- Seeds: PF-02, V3-PF-007, EXP-029

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

- Pareto front size: 49
- Hypervolume: 0.875
- Convergence: false
- Best operating region remains non-converged, supporting the Phase 5 transition

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
