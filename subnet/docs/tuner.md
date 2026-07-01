# Tuner

## NSGA-II configuration

- Population: 30
- Generations: 20
- Objectives: 4
- Crossover: SBX, probability 0.9, eta 15
- Mutation: polynomial mutation, eta 20
- Active seed lineage: EXP-116, EXP-118, EXP-140, EXP-141
- NSGA-II profile: v13 R3
- Evaluation path: `InsigniaTuningProblem._evaluate` (`subnet/tuning/optimizer.py:172-213`) calls `SimulationHarness.run()` directly per candidate — no surrogate. `compute_fitness` (`optimizer.py:88-112`) reads `sim_result.adversarial_researcher_scores` from the harness output.
- Sampling: Latin Hypercube + elite seeding
- Elite seeds injected: EXP-140, EXP-141, EXP-134, EXP-132, EXP-133, EXP-135

> **Correction (2026-07-01).** Earlier versions of this file listed "Surrogate
> model: Gaussian Process, trained from 93 empirical data points / Surrogate
> fit: R2 = 0.96 / Active surrogate variable count: 41." A code audit found
> no GP surrogate, no `sklearn.gaussian_process` import, no `surrogate.predict`
> anywhere in the codebase. The "93 empirical points / R²=0.96 / 41 vars"
> figures appear only as metadata in `NSGA2_V13_PROFILE` (`optimizer.py:64-85`)
> and in `reference_configs/knee_point_V13-R3.json`; they do not describe any
> code path. The optimizer is a plain NSGA-II over the harness.

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
- Harness-measured knee (`V13-R3-KP-020-a3c7`) — **separation falsified by reproduction**:
  - breach_rate: `2.6e-6` (unconfirmed)
  - honest_score: `0.9808` (empirical `~0.977`, holds)
  - separation: `0.963` ❌ (empirical `~0.23`; `tests/test_simulation_separation.py` reproduces 0.15-0.22 with current code)
  - variance: `0.00081`
- Root cause (corrected 2026-07-01, see EMULATOR_SPEC §6.6): the harness at `simulation.py:870-873` applies an anti-copy multiplier only to `CopycatMiner` and `CopyTrader`. `SybilMiner`, `OverfittingMiner`, `SingleMetricGamer`, and `PartnerGamer` have no penalty path and score ~0.90 — `SybilMiner` scores *higher* than honest (0.9163 vs 0.9151) because `sybil_pressure` / `ensemble_signals` are computed but never applied to `miner_scores`. The earlier "GP surrogate trained on biased analytical adversary scores" explanation described a surrogate that does not exist in code.
- Required fix before any future knee is trusted: add penalty paths in `simulation.py:870-875` for every adversary type enumerated in EMULATOR_SPEC §5.1/§5.2. The `test_no_adversary_outscores_honest_mean` test in `tests/test_simulation_separation.py` pins this — no adversary type may score higher than the honest mean. Re-validate separation on-chain after the harness gate clears.
- Last non-contradicted checkpoint: `V13-R2-KP-020-a7f2` (breach `3.5e-6`, honest `0.9795`, separation `0.953`) — ⚠️ researcher memory key `emulator_spec_execution_state` notes this checkpoint was scored under the same incomplete-harness code path and may also fail empirical re-validation. Treat as provisional until re-validated.
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

- ⚠️ **Correction (2026-07-01):** an earlier version of this section attributed the V13-R3 separation failure to a "GP surrogate trained on biased analytical adversary scores." A code audit found no surrogate exists in `subnet/tuning/optimizer.py` — the optimizer calls `SimulationHarness.run()` directly. The binding open failure is still the separation gate (empirical `~0.23` vs gate `≥0.90`), but the root cause is incomplete anti-gaming penalty coverage in the harness (`simulation.py:870-873` penalizes only Copycat/CopyTrader; sybil, overfitter, single_metric_gamer, and partner_gamer have no penalty path). See EMULATOR_SPEC §6.6 (corrected) and `tests/test_simulation_separation.py`.
- **Highest-priority fix:** add penalty paths in `simulation.py:870-875` for every adversary type enumerated in EMULATOR_SPEC §5.1/§5.2. The `test_no_adversary_outscores_honest_mean` test pins this. Tightening only the Copycat/CopyTrader multiplier (already done, 0.50 → 0.10) is insufficient — separation rose from 0.15 to 0.22, still well below 0.90.
- Commit-reveal remains a validated prerequisite, with system-level effectiveness `0.76` and simulator-estimated pre/post effectiveness `0.801`.
- The active knee-point defense stack is:
  - decentralized identity verification (`identity_bond = 0.08`)
  - stake-based consensus
  - Bayesian model averaging
  - commit-reveal
- `PC-VH-006` is now deployed and should be treated as part of the active tuning baseline rather than a future recommendation.
- Remaining work is empirical:
  - verify the `2.6e-6` breach rate in live (online) simulation once the local chain at `ws://10.0.0.193:9945` is reachable (currently `ECONNREFUSED` — see deployer_state)
  - measure realized Sybil reduction from `0.195` toward the projected `0.08`
