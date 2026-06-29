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

> ⚠️ **V13-R3 INVALIDATED (Orchestration Report 2026-06-29T01-35-48).** The
> surrogate knee `V13-R3-KP-020-a3c7` predicted separation `0.963`, but empirical
> validation against the full adversarial population measured `~0.23` — the best
> adversary (Copycat) scores `0.733` and adversaries capture ~64.7% of chain
> weight. V13-R3 **fails** the `≥0.90` separation gate and is **not promoted**.
> The last non-contradicted checkpoint is the R2 knee.

- Configured search regime: 20 generations x 30 population
- Surrogate-predicted knee (`V13-R3-KP-020-a3c7`) — **predictions only, separation falsified**:
  - breach_rate: `2.6e-6` (unconfirmed)
  - honest_score: `0.9808` (empirical `~0.977`, holds)
  - separation: `0.963` ❌ (empirical `~0.23`)
  - variance: `0.00081`
- Root cause: the optimizer's internal analytical adversary model under-scored Copycat/CopyTrader (`~0.02-0.05` vs `~0.73` empirical), so the GP surrogate (`R²=0.96` to *biased* targets) optimized a false objective; the Pareto front collapsed into a narrow `0.958-0.967` separation band (false local optimum)
- Required fix before any future knee is trusted: feed **empirical** (harness) adversary scores back into surrogate training; re-validate separation on-chain
- Last non-contradicted checkpoint: `V13-R2-KP-020-a7f2` (breach `3.5e-6`, honest `0.9795`, separation `0.953`)
- Full reference + empirical results: [../reference_configs/knee_point_V13-R3.json](../reference_configs/knee_point_V13-R3.json)
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

- ⚠️ **Correction:** an earlier run *appeared* to confirm the surrogate could beat the `5e-6` target, but empirical validation (Orchestration Report 2026-06-29) showed the surrogate's adversary scores were biased low, so its breach/separation predictions are unreliable. **Do not treat surrogate-predicted gate passes as achieved.** The binding open failure is the separation gate (empirical `~0.23` vs gate `≥0.90`).
- **Highest-priority fix:** the optimizer/surrogate must be trained against the empirical (simulation-harness) adversary scores, not the internal analytical adversary model that under-scores Copycat/CopyTrader. Until then, NSGA-II will keep converging to false optima.
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
