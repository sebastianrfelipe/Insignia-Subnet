# Insignia Subnet - Agent Swarm Protocol

This document is the MCP swarm prompt for the Insignia subnet repository.

---

## 1. Current system state

- Current phase: **Phase 4 - Attack Surveillance**
- Transition status: **Phase 5 transition viable**
- Latest reported generation: **58**
- Latest checkpoint metrics:
  - breach_rate: `0.0008`
  - honest_score: `0.935`
  - score_separation: `0.75`
  - hypervolume: `0.875`
  - Pareto front size: `49`
- Best reported experiment: **EXP-116**
  - breach_rate: `0.000049`
  - honest_score: `0.9705`
- Commit-reveal validation status: **validated**
  - effectiveness: `0.72`
  - validator latency severity: `0.043` headline / `0.033-0.043` observed band
  - prediction timing severity: `0.016-0.025`
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
- `validator_latency_penalty_weight = 0.25`
- `high_latency_threshold_ms = 2000`
- `commit_rate_threshold = 0.70`
- `commitment_violation_weight = 0.008`
- `selective_reveal_warning_streak = 1`
- `selective_reveal_penalty_streak = 2`
- `selective_reveal_zero_streak = 3`

### Ensemble detection defaults

- `correlation_threshold = 0.77`
- `entropy_threshold_lower = 0.18`
- `symbol_diversity_threshold = 0.275`
- `fusion_strategy = weighted_voting_dynamic_adaptive`
- `response_vote_threshold = 2`

### NSGA-II defaults

- population: `30`
- generations: `50`
- objectives: `4`
- SBX crossover: `prob=0.9`, `eta=15`
- polynomial mutation: `eta=20`
- seed lineage: `PF-02`, `V3-PF-007`, `EXP-029`

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

### Report-driven surveillance extensions now implemented

20. selective_revelation
21. statistical_anomaly
22. behavioral_anomaly
23. temporal_attack_pattern
24. sybil_collusion_graph
25. cross_layer_correlation

### Highest-priority persistent risk

The main unresolved structural warning remains Sybil pressure caused by market
concentration. Swarm agents should treat pair diversification as a permanent
defense objective, not a one-off patch.

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

Discard when:
- breach rate increases materially, or
- scalarized fitness regresses

### TSV schema

The researcher writes TSV rows with:

- `commit`
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
- `EXP-116` - current best overall
- `EXP-118` - Phase 5 transition checkpoint

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

1. validate commit-reveal effectiveness above the `0.667` floor under repeated simulation
2. continue reducing the persistent Sybil warning tied to pair imbalance
3. use `EXP-116` and `EXP-118` as the main operating references for next-cycle tuning
4. keep `program.md` synchronized with actual repo behavior

---

## 13. Hard rules for the swarm

1. do not remove the 10-metric L2 scorer to fit a smaller summary table
2. do not reintroduce BTC-only assumptions in new simulation code
3. do not let `program.md` drift from actual repo behavior
4. prefer the detector names in code over older historical numbering in archived notes
5. when docs and code conflict, update docs to match code unless a deliberate migration is underway
6. continue until interrupted; do not pause to ask for permission
