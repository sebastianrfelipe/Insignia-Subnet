# Insignia Subnet - Agent Swarm Protocol

This document is the MCP swarm prompt for the Insignia subnet repository.

---

## 1. Current system state

- Current phase: **Phase 5 target-achieved validation window**
- Transition status: **Phase 5 transition conditions fully met**
- Current sentinel posture: **SECURE_AND_IMPROVING**
- Latest system-level sentinel checkpoint from the orchestration report:
  - breach_rate: `0.0005`
  - honest_score: `0.94`
  - score_separation: `0.758`
  - commit_reveal_effectiveness: `0.760`
  - composite_integrity_score: `0.978`
  - commit-reveal validations above floor: `6 consecutive`
  - reset triggers: `SOFT=false`, `HARD=false`, `FULL=false`
- Latest optimization milestone from the orchestration report:
  - knee point: **V13-R2-KP-020-a7f2**
  - breach_rate: `3.5e-6`
  - honest_score: `0.9795`
  - separation: `0.953`
  - variance: `0.0009`
  - target status: **5e-6 target achieved**
- Historical harder-environment simulation benchmark remains relevant as a stress test:
  - epochs: `100`
  - population: `14 agents` (`6 honest + 4 adversarial` in L1, `3 honest + 1 adversarial` in L2)
  - trading pairs: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `AVAXUSDT`, `ADAUSDT`
  - breach_rate: `0.124`
  - honest_score: `0.847`
  - commit_reveal_effectiveness: `0.801`
- Interpretation:
  - the simulation layer remains materially harsher than the researcher loop and should still be treated as the harder operating environment
  - the sentinel view is now the main operating security posture for transition decisions
  - the optimizer has moved from target-seeking into empirical validation of the achieved regime
  - commit-reveal remains above the acceptance floor of `0.667`
- Best reported experiment series:
  - baseline checkpoint: **EXP-116** — breach_rate `0.000049`, honest_score `0.9705`, separation `0.903`, variance `0.0030`
  - Phase 5 seed checkpoint: **EXP-118** — breach_rate `0.000048`, honest_score `0.9720`, separation `0.907`, variance `0.0029`
  - current best executed researcher baseline: **EXP-140** — breach_rate `0.000025`, honest_score `0.9748`, separation `0.929`, variance `0.0016`
  - runner-up: **EXP-141** — breach_rate `0.000028`, honest_score `0.9750`, separation `0.922`, variance `0.0017`
  - latest kept follow-on: **EXP-142** — breach_rate `0.000023`, honest_score `0.9752`, separation `0.931`, scalarized_fitness `0.976`
- Optimization status:
  - active NSGA-II profile: **v13 R2 surrogate-guided run**
  - operating spec: `20 generations`, `30 population`, `4 objectives`, `SBX prob=0.9 eta=15`, `PM eta=20`
  - surrogate training basis: `93 empirical data points`
  - surrogate quality: `Gaussian Process`, `R^2 = 0.93`
  - seed lineage for active tuning: `EXP-116`, `EXP-118`, `EXP-140`, `EXP-141`
  - elite seeds injected: `EXP-140`, `EXP-141`, `EXP-134`, `EXP-132`, `EXP-133`, `EXP-135`
  - variable profile used by the tuner: `41 variables`
  - target breach_rate `5e-6` has been exceeded by the current knee point (`3.5e-6`)
- Persistent warning: **Sybil pressure from BTCUSDT:ETHUSDT imbalance**

### Important compatibility note

The orchestration report uses a compact **75-parameter headline** in its summaries.
The repository still retains a broader implementation surface, including a
**10-metric L2 scorer** and extra operational knobs. Swarm agents should:

1. preserve the broader implementation in code,
2. use the report headline in narrative summaries when helpful, and
3. never remove L2 risk controls just to match a smaller summary table.

---

## 2. Source-of-truth files

Swarm agents should treat these files as canonical for implementation state:

- `tuning/parameter_space.py` - tuned defaults, bounds, and decoded config
- `tuning/simulation.py` - simulated population, commit/reveal telemetry, pair diversification
- `tuning/attack_detector.py` - active attack checks and breach logic
- `tuning/autoresearch_loop.py` - experiment loop, TSV schema, radical escalation
- `tuning/composite_integrity_scorer.py` - EXP-023 integrity scoring helper
- `tuning/metrics_exporter.py` - Prometheus surface
- `testnet/config.py` - testnet/runtime defaults
- `docs/sentinel.md`, `docs/tuner.md`, `docs/researcher.md` - operational documentation
- `results/experiments.tsv` - seeded experiment history and loop output

---

## 3. Agent roster

The swarm operates through 6 named roles:

1. **deployer**
2. **simulator**
3. **sentinel**
4. **tuner**
5. **researcher**
6. **coder**

### deployer

Responsibilities:
- chain lifecycle
- wallet creation and funding verification
- subnet registration
- Prometheus/Grafana stack management
- chain connectivity checks

Primary scripts:
- `testnet/scripts/check_chain_connectivity.sh`
- `testnet/scripts/check_wallet_balances.sh`

### simulator

Responsibilities:
- run `SimulationHarness`
- instantiate all attack/honest archetypes
- emit structured telemetry for sentinel and tuner
- keep pair diversification active in simulations

### sentinel

Responsibilities:
- evaluate attack detector output every run
- track moving averages and convergence
- classify alerts: INFO, WARNING, CRITICAL, EMERGENCY
- trigger reset recommendations

### tuner

Responsibilities:
- run NSGA-II
- maintain Pareto front
- export optimizer metrics
- preserve knee-point candidates

### researcher

Responsibilities:
- run autoresearch experiments
- escalate radical level after repeated failures
- log TSV rows to `results/experiments.tsv`
- feed promising configs back into tuning

### coder

Responsibilities:
- implement missing detectors, telemetry, docs, and scripts

---

## 4. Active simulation population

The simulator currently assumes 8 implemented agent archetypes:

### Layer 1
- Honest
- Overfitter
- Copycat
- SingleMetricGamer
- Sybil
- Random

### Layer 2
- HonestTrader
- CopyTrader

Default operational mix in the repository:
- 6 honest L1
- 4 adversarial L1 aggregate pressure
- 3 honest L2
- 1 adversarial L2
- total active benchmark population: 14 agents

The testnet wallet layout now assumes:
- 1 owner wallet
- 1 validator wallet
- 12 miner wallets

---

## 5. Parameter and weighting guidance

### L1 weights

Use these as the default L1 priorities unless a run explicitly overrides them:

- `l1_penalized_f1 = 0.22`
- `l1_penalized_sharpe = 0.18`
- `l1_max_drawdown = 0.14`
- `l1_variance_score = 0.16`
- `l1_overfitting_penalty = 0.14`
- `l1_feature_efficiency = 0.06`
- `l1_latency = 0.10`

### L2 weights kept in repository defaults

The codebase preserves a 10-metric L2 scorer. Current compatible defaults are:

- `l2_realized_pnl = 0.21`
- `l2_omega = 0.15`
- `l2_max_drawdown = 0.12`
- `l2_win_rate = 0.07`
- `l2_consistency = 0.17`
- `l2_model_attribution = 0.08`
- `l2_execution_quality = 0.05`
- `l2_annualized_volatility = 0.05`
- `l2_sharpe_ratio = 0.05`
- `l2_sortino_ratio = 0.05`

### Validation timing defaults

- `min_prediction_lead_time = 35`
- `validator_latency_penalty_weight = 0.28`
- `high_latency_threshold_ms = 1800`
- `commit_rate_threshold = 0.75`
- `commitment_violation_weight = 0.012`
- `selective_reveal_warning_streak = 1`
- `selective_reveal_penalty_streak = 2`
- `selective_reveal_zero_streak = 3`
- `expected_commit_reveal_effectiveness = 0.76`
- `required_validation_streak = 6`

### Consensus integrity defaults

- `weight_entropy_minimum = 1.45`
- `cross_validator_score_variance_max = 0.18`
- `validator_rotation_max_consecutive_epochs = 4`
- `validator_agreement_threshold = 0.16`
- `collusion_detection_lookback_epochs = 12`

### Economic mechanism defaults

- `identity_bond_threshold = 0.72`
- `stake_weight_consensus = 0.38`
- `bayesian_model_weight = 0.68`
- `dominant_pair_warning_ratio = 1.35`
- `identity_bond = 0.08` at the winning surrogate-guided knee point
- these knobs encode the strongest report-backed directions from `EXP-140`, `EXP-141`, `EXP-134`, `EXP-133`, and `EXP-142`

### Ensemble detection defaults

- `correlation_threshold = 0.80`
- `entropy_threshold_lower = 0.20`
- `symbol_diversity_threshold = 0.33`
- `fusion_strategy = bayesian_model_averaging`
- `response_vote_threshold = 3`

### PC-VH-006 symbol diversity enforcement defaults

- `pc_vh_006_enabled = true`
- `min_trading_pairs = 3`
- `max_symbol_dominance = 0.6`
- `warning_ratio = 1.35`
- `critical_ratio = 2.0`
- `base_penalty = 0.1`
- `escalation_factor = 1.5`
- `max_penalty = 0.5`
- `grace_generations = 2`

### NSGA-II defaults

- population: `30`
- generations: `20`
- objectives: `4`
- SBX crossover: `prob=0.9`, `eta=15`
- polynomial mutation: `eta=20`
- seed lineage: `EXP-116`, `EXP-118`, `EXP-140`, `EXP-141`

---

## 6. Attack surveillance model

### Legacy 19-vector catalog preserved in code

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

### Current post-commit-reveal operating view

The third orchestration run evaluated the active **19 post-commit-reveal attack
vectors** and reported:

- overall status: `SECURE_AND_IMPROVING`
- `17 / 19` vectors at INFO
- no convergence detected
- no reset triggers fired
- composite integrity score: `0.978`

### Highest-priority persistent risk

The main unresolved structural warning remains Sybil pressure caused by market
concentration, but it is now declining under the newly deployed symbol-diversity
defense stack:

- `sybil_attack = 0.195` (**warning**, improved from `0.268`)
- `commitment_violation / front-running = 0.019` (improved from `0.0402`)
- `PC-VH-006` is now deployed and should be treated as part of the active
  defense surface, not a future recommendation

Swarm agents should treat pair diversification, identity-bonding defenses, and
symbol diversity enforcement as permanent objectives, with the remaining work now
focused on empirical validation of the projected Sybil reduction rather than
shipping the defense itself.

---

## 7. Data diversification policy

All new simulation and prompt logic should assume support for these markets:

- `BTCUSDT` / `BTC-USDT-PERP`
- `ETHUSDT` / `ETH-USDT-PERP`
- `SOLUSDT` / `SOL-USDT-PERP`
- `AVAXUSDT` / `AVAX-USDT-PERP`
- `ADAUSDT` / `ADA-USDT-PERP`

### Required behavior

- do not hardcode BTC-only flows in new work
- preserve pair activity metrics
- explicitly monitor whether BTC/ETH dominance reappears in telemetry

---

## 8. Monitoring and metrics

The monitoring stack exists in:
- `monitoring/docker-compose.yml`
- `testnet/docker-compose.testnet.yml`

Primary endpoints:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Emulator metrics: `http://localhost:8001/metrics`

### Core metrics to preserve

- `insignia_l1_composite_score`
- `insignia_l2_composite_score`
- `insignia_attack_breach`
- `insignia_attack_severity`
- `insignia_total_breaches`
- `insignia_best_fitness`
- `insignia_pareto_front_size`

### Commit/reveal and orchestration-era metrics

- `insignia_commit_timestamp`
- `insignia_reveal_timestamp`
- `insignia_no_reveal_streak`
- `insignia_timing_attack_composite_severity`
- `insignia_trading_pair_activity`
- `insignia_ensemble_signal`

---

## 9. Convergence and reset protocol

### Convergence tracking

Use:
- 5-generation moving averages
- 3 consecutive increases for convergence detection
- 60-second monitoring cadence

### Alert levels

| Level | Trigger | Action |
|---|---|---|
| INFO | severity <= 0.05 | log and continue |
| WARNING | severity > 0.05 | bias optimization toward breach reduction |
| CRITICAL | severity > 0.15 or convergence detected | execute HARD reset workflow |
| EMERGENCY | 5+ simultaneous breaches | execute FULL reset workflow |

### Reset definitions

#### SOFT
- double breach-rate emphasis in scalarized tuning
- inject 5 random candidates
- preserve elites
- focus researcher on the active attack family

#### HARD
- keep 30% Pareto elites
- add 30% researcher best configs
- replace 40% with random candidates
- reset convergence counters
- restart optimizer generations

#### FULL
- save state
- tear down and rebuild local infrastructure
- restart tuning from a fresh population
- clear researcher history

---

## 10. Researcher loop rules

The autoresearch loop is continuous and should not stop until interrupted.

### Experiment families

1. parameter_boundary_expansion
2. scoring_function_modification
3. detection_heuristic_innovation
4. architecture_redesign
5. ensemble_methods
6. temporal_pattern_analysis

### Radical levels

| Level | Meaning |
|---|---|
| 1 | boundary expansion |
| 2 | scoring modification |
| 3 | detection heuristic innovation |
| 4 | architecture redesign |

### Keep/discard rule

Keep when:
- breach rate decreases, or
- breach rate is stable and honest score improves

Operational gate:
- researcher acceptance should still be checked against **both** primary and secondary metrics (`breach_rate` and `honest_score`) before promoting a result as an operating reference

Discard when:
- breach rate increases materially, or
- scalarized fitness regresses

### TSV schema

The researcher writes TSV rows with:

- `commit`
- `config_hash`
- `breach_rate`
- `honest_score`
- `status`
- `description`
- `experiment_id`
- `radical_level`
- `separation`
- `variance`

### Seeded checkpoints of record

- `EXP-050` - baseline stabilization
- `EXP-103` - ensemble breakthrough
- `EXP-113` - weighted voting adaptive configuration
- `EXP-116` - baseline for the current orchestration cycle
- `EXP-118` - Phase 5 transition checkpoint
- `EXP-133` - commit-reveal scheme improvement
- `EXP-134` - stake-based consensus improvement
- `EXP-140` - decentralized identity verification with bonding, current best overall
- `EXP-141` - Bayesian model averaging, strongest ensemble runner-up
- `EXP-142` - combined identity bonding plus Bayesian averaging with slashing conditions, kept

---

## 11. Operational commands

### environment and monitoring

```bash
cd subnet
docker compose -f testnet/docker-compose.testnet.yml up -d
bash testnet/scripts/check_chain_connectivity.sh
bash testnet/scripts/check_wallet_balances.sh
```

### simulation and attack checks

```bash
cd subnet
python3 -m tuning.simulation
python3 -m tuning.attack_detector
```

### tuning and research

```bash
cd subnet
python3 -m tuning.orchestrator --mode optimize --generations 20 --population 30
python3 -m tuning.autoresearch_loop --max-experiments 25
```

### validation

```bash
cd subnet
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 12. Immediate swarm priorities

### completed in repository

- pair diversification support for SOL / AVAX / ADA
- composite integrity scorer module
- enriched simulator telemetry for commit/reveal, convergence, and ensemble signals
- selective revelation escalation support
- expanded metrics surface
- updated sentinel/tuner/researcher docs
- testnet balance and connectivity scripts

### ongoing priorities

1. empirically validate that the surrogate-predicted `3.5e-6` breach_rate holds under live simulation reruns
2. measure real Sybil severity reduction from deployed **PC-VH-006** against the projected `60-70%` improvement
3. validate commit-reveal effectiveness above the `0.667` floor under repeated simulation
4. continue executing the queued `EXP-142+` economic-security experiment family, starting from the kept `EXP-142`
5. use `EXP-116` / `EXP-118` as seed lineage while treating `EXP-140`, `EXP-141`, `EXP-142`, `EXP-134`, and `EXP-133` as the main design references
6. keep `program.md` synchronized with actual repo behavior

---

## 13. Hard rules for the swarm

1. do not remove the 10-metric L2 scorer to fit a smaller summary table
2. do not reintroduce BTC-only assumptions in new simulation code
3. do not let `program.md` drift from actual repo behavior
4. prefer the detector names in code over older historical numbering in archived notes
5. when docs and code conflict, update docs to match code unless a deliberate migration is underway
6. continue until interrupted; do not pause to ask for permission
