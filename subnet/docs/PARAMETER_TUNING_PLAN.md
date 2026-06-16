# Insignia Subnet — Parameter Tuning Strategy

> **Two NSGA-II layers — do not confuse them.** This document covers the
> **OFFLINE mechanism tuner** (`tuning/optimizer.py`), which searches *mechanism
> parameters* (scoring weights, thresholds, and the `pairing` group). It is
> distinct from the **in-protocol** genetic algorithm in `insignia/pairing.py`,
> which evolves `(researcher, trader)` pairings each epoch on-chain. Both use
> NSGA-II but at different layers. See [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md).

## Executive Summary

The Insignia subnet's incentive mechanism combines a **model scoring vector** (7 metrics) and a **trading scoring vector** (10 metrics) into a joint pair fitness, plus **pairing parameters**, **anti-gaming thresholds**, **commit-reveal parameters**, **emission distribution parameters**, and **subnet hyperparameters**. The full tuning landscape spans two distinct levels:

- **Insignia application-level parameters** — scoring weights, thresholds, commit-reveal timing, defense parameters, emission distribution, and the pairing/genetic knobs (tuned by the emulator)
- **Bittensor on-chain subnet hyperparameters** — network-level parameters controlling registration, consensus, staking, and bonds (set via `btcli subnets hyperparameters`)

Tuning by hand is infeasible because:

1. The parameter space is high-dimensional across two levels
2. Interactions between parameters are non-linear (e.g., changing the overfitting weight affects which models win pairs; changing `tempo` affects how frequently weights are set)
3. Attack vectors exploit specific parameter configurations
4. You cannot iterate quickly on mainnet — miners are real actors with real stakes

**Solution:** Deploy on Bittensor testnet, replace real miners with AI agent bots that simulate honest, adversarial, and degenerate strategies, instrument everything with Prometheus/Grafana, and use evolutionary multi-objective optimization (pymoo NSGA-II) to search the parameter space automatically.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│  Manages simulation lifecycle, parameter injection,         │
│  fitness evaluation, and evolutionary search loop            │
├─────────────┬───────────────┬───────────────┬───────────────┤
│  Parameter  │  Simulation   │  Attack       │  Evolutionary │
│  Space      │  Harness      │  Detector     │  Optimizer    │
│  Definition │  (AI Agents)  │               │  (pymoo)      │
└──────┬──────┴───────┬───────┴───────┬───────┴───────┬───────┘
       │              │               │               │
       │              ▼               │               │
       │   ┌──────────────────────┐   │               │
       │   │ Subnet Pipeline      │   │               │
       │   │ Researcher bots      │   │               │
       │   │ Trader bots          │◄──┘               │
       │   │ PairedValidator      │                   │
       │   │  (pairing + NSGA-II  │                   │
       │   │   + marginal credit) │                   │
       │   └────────┬─────────────┘                   │
       │            │                                 │
       │            ▼                                 │
       │   ┌─────────────────┐                        │
       │   │   Prometheus    │                        │
       │   │   Metrics       │──────────────────────►│
       │   └────────┬────────┘     fitness scores     │
       │            │                                 │
       │            ▼                                 │
       │   ┌─────────────────┐                        │
       │   │    Grafana      │                        │
       │   │   Dashboards    │                        │
       │   └─────────────────┘                        │
       │                                              │
       └──────────────────────────────────────────────┘
              Parameter configs injected per generation
```

---

## Step-by-Step Execution Plan

### Phase 1: Parameter Space Definition

**Goal:** Enumerate every tunable parameter, define its bounds, and categorize by layer.

#### Insignia Application-Level Parameters (68 total, tuned by emulator)

| Category | Parameters |
|----------|-----------|
| Model Scoring Weights | penalized_f1, penalized_sharpe, max_drawdown, variance_score, overfitting_penalty, feature_efficiency, latency |
| Trading Scoring Weights | realized_pnl, omega, max_drawdown, win_rate, consistency, model_attribution, execution_quality, annualized_volatility, sharpe_ratio, sortino_ratio |
| Overfitting Detector | gap_threshold, decay_rate |
| **Pairing (genetic mechanism)** | partners_per_miner, elite_fraction, mutation_rate, pair_blend_alpha, marginal_contribution_weight, fixed_pair_correlation_threshold, max_pairs |
| Anti-Gaming | fingerprint_correlation_threshold, copy_trade_time_tolerance, copy_trade_size_tolerance, copy_trade_correlation_threshold |
| Trading Engine | base_spread_bps, volatility_impact_factor, size_impact_factor, fee_bps, max_position_pct, max_drawdown_pct |
| Buyback | buyback_pct, min_profit_threshold |
| Emission Distribution | sigmoid_midpoint, sigmoid_steepness |
| Rate Limiting | rate_limit_epoch_seconds |
| Commit-Reveal Timing | commit_window_seconds, reveal_window_seconds, late_reveal_penalty |
| Validation Timing | min_prediction_lead_time, validator_latency_penalty_weight, high_latency_threshold_ms, commit_rate_threshold, commitment_violation_weight, selective_reveal_warning_streak, selective_reveal_penalty_streak, selective_reveal_zero_streak |
| Consensus Integrity | weight_entropy_minimum, cross_validator_score_variance_max, validator_rotation_max_consecutive_epochs, validator_agreement_threshold, collusion_detection_lookback_epochs |
| Economic Mechanisms | identity_bond_weight, identity_bond_threshold, stake_weight_consensus |
| Ensemble Detection | bayesian_model_weight |
| Market Data | dominant_pair_warning_ratio + PC-VH-006 symbol-diversity knobs |

> The legacy `promotion_*` / `feedback_*` groups (top_n, min_consecutive_epochs,
> bonus/penalty weights, etc.) and the cross-layer split (`l1_l2_emission_split`,
> `cross_layer_penalty_strength`, `cross_layer_latency`) were removed; the
> `pairing` group replaces them. Exact bounds and defaults live in
> `tuning/parameter_space.py`.

#### Bittensor On-Chain Subnet Hyperparameters (39 total, set via btcli)

| Category | Parameters | Count |
|----------|-----------|-------|
| Core | rho, kappa, tempo, immunity_period, min_allowed_weights, max_weight_limit, max_validators | 7 |
| Difficulty & Registration | difficulty, min_difficulty, max_difficulty, min_burn, max_burn, registration_allowed, target_regs_per_interval, max_regs_per_block, adjustment_interval | 9 |
| Weights & Adjustments | weights_version, weights_rate_limit, adjustment_alpha, activity_cutoff | 4 |
| Commit-Reveal | commit_reveal_weights_enabled, commit_reveal_period | 2 |
| Alpha & Staking | alpha_high, alpha_low, alpha_sigmoid_steepness, liquid_alpha_enabled | 4 |
| Bonds | bonds_moving_avg, bonds_reset_enabled | 2 |
| Rate Limits | serving_rate_limit | 1 |
| Network State | yuma_version, subnet_is_active, transfers_enabled, user_liquidity_enabled | 4 |
| **Subtotal** | | **33** |

> **Note:** The Bittensor SDK `SubnetHyperparameters` dataclass enumerates these fields. The exact count may vary by SDK version as the protocol evolves. The 6 most operationally relevant subnet hyperparameters (tempo, immunity_period, min_allowed_weights, max_weight_limit, adjustment_alpha, bonds_moving_avg) are the primary targets for subnet-owner tuning.

#### Full Tuning Landscape

| Level | Tuned By |
|-------|----------|
| Insignia application-level (scoring weights, thresholds, pairing group, defenses) | Offline NSGA-II tuner (`tuning/optimizer.py`) |
| Bittensor on-chain hyperparameters | btcli / subnet owner configuration |

> Note: the offline tuner tunes the *rules*; the in-protocol genetic algorithm
> (`insignia/pairing.py`) selects `(researcher, trader)` pairs under those rules
> each epoch.

**Constraints:**
- Model weights must sum to 1.0
- Trading weights must sum to 1.0
- All weights ∈ [0.01, 0.50]
- Thresholds must be positive
- `partners_per_miner` ≥ 2 (every miner judged against multiple partners)

**Implementation:** `tuning/parameter_space.py` — defines the search space, encoding/decoding, and constraint handling.

---

### Phase 2: Simulation Harness with AI Agent Miners

**Goal:** Build a self-contained simulation that runs the full subnet pipeline with configurable bot miners instead of real network participants.

#### Miner Agent Types

| Agent Type | Role | Behavior | Purpose |
|------------|------|----------|---------|
| **Honest** | researcher | Trains models normally, submits best effort | Baseline — good-faith researchers |
| **Overfitter** | researcher | Deliberately overfits to training data | Tests overfitting detection |
| **Copycat** | researcher | Copies another miner's model with small perturbations | Tests plagiarism detection |
| **Single-Metric Gamer** | researcher | Optimizes only for directional accuracy | Tests composite scoring robustness |
| **Sybil** | researcher | Multiple identities submitting correlated models | Tests sybil detection |
| **Random** | researcher | Random model submissions | Noise floor baseline |
| **HonestTrader** | trader | Builds a strategy on the assigned model | Baseline — good-faith traders |
| **CopyTrader** | trader | Mirrors another trader's positions | Tests copy-trade detection |
| **ColludingResearcher + ColludingTrader** | both | A ring that only performs when matched together | Tests `pair_collusion` + marginal credit |
| **PartnerGamingTrader** | trader | Tries to steer which partner it is matched with | Tests `partner_selection_gaming` |

Each agent type is parameterized so the optimizer can test different attack intensities.

#### Model-route diversity during tuning

To better approximate decentralized intelligence diversity, the tuning harness can
assign each simulated miner/trader agent a stable per-run external model route.
This is intended for MCP-backed routing environments where the backend already
supports route selection.

Key properties:
- route assignment is **stable per miner for the full run**
- assignments are seeded and reproducible
- assignments are emitted into simulation and emulator telemetry
- route diversity is treated as a realism/control variable, not as untracked noise

**Implementation:** `tuning/simulation.py` — contains `MinerAgent` classes, `SimulationHarness`, and the full pipeline runner.

---

### Phase 3: Attack Vector Detection

**Goal:** Automatically detect whether each documented attack vector has been breached under a given parameter configuration. `tuning/attack_detector.py` evaluates 28 vectors, including the 3 paired-mechanism vectors below.

For each attack vector, define a **breach condition** (boolean) and a **severity score** (0-1):

| # | Attack | Breach Condition |
|---|--------|-----------------|
| 1 | Overfitting exploitation | Overfitting miner scores higher than honest miners |
| 2 | Submission spam | Spammer gets >0 score despite rate limits |
| 3 | Model plagiarism (researcher) | Copycat miner is not detected and scores independently |
| 4 | Copy-trading (trader) | Copy-trader is not detected and scores independently |
| 5 | Single-metric gaming | Single-metric gamer ranks in top 50% |
| 6 | Validator data leakage | (Structural check — no simulation needed) |
| 7 | Paper trading manipulation | Inflated P&L not caught by slippage model |
| 8 | Sybil attack | Sybil identities collectively earn >2x single miner share |
| 9 | Regime-specific exploitation | Model that only works in 1 regime ranks in top 25% |
| 10 | Pair collusion | A researcher+trader ring scores well only when matched together (non-transferable lift) |
| 11 | Partner-selection gaming | A miner steers which partner it is matched with to secure a favorable counterpart |
| 12 | Objective weight manipulation | NSGA-II objective weights manipulated to favor specific attack patterns |
| 13 | GA parameter exploitation | Genetic algorithm parameters (crossover, mutation) tuned to exploit optimizer |
| 14 | Governance parameter manipulation | On-chain hyperparameters manipulated to enable other attacks |
| 15 | Latency arbitrage in pairing | Exploiting validator latency or partner foreknowledge to submit after data materializes |
| 16 | Pareto front manipulation | Population diversity or elite preservation exploited to bias optimization |
| 17 | Reward distribution manipulation | Sigmoid emission curve or buyback parameters gamed for disproportionate rewards |
| 18 | Validator latency exploitation (Vector 8) | Miner accuracy correlated with validator latency; post-market submissions scored as predictions. **Commit-reveal validated: projected severity 0.047 < 0.05 target** |
| 19 | Miner-validator collusion | Validator weight/score bias toward specific miners not justified by performance |

**Implementation:** `tuning/attack_detector.py` — runs the simulation with adversarial agents and returns a breach report across 28 vectors (the legacy post-commit-reveal set plus `pair_collusion`, `partner_selection_gaming`, and `latency_arbitrage_pairing`, with the former L1/L2-skew vector reframed as role-emission balance).

**Commit-reveal impact on Vector 18:** Sentinel validation (session 69dab601) confirmed that the commit-reveal scheme reduces Vector 18 severity from 0.09 to projected 0.047, clearing the 0.05 target. Companion Vector 11 (prediction timing manipulation) also drops from 0.06 to 0.025. The critical effectiveness threshold is 0.667 — this must be validated in simulation before production deployment.

---

### Phase 4: Prometheus Metrics & Grafana Dashboards

**Goal:** Instrument the simulation with metrics so you can visualize tuning progress in real-time.

#### Metrics Exported

**Simulation Metrics:**
- `insignia_model_composite_score` (gauge, per researcher, per generation)
- `insignia_pair_composite_score` (gauge, per pair)
- `insignia_trading_composite_score` (gauge, per trader)
- `insignia_miner_weight` (gauge, per miner — single emission vector)
- `insignia_pareto_front_size` (gauge, per generation)
- `insignia_pair_count` (gauge, per generation)
- `insignia_collusion_flags` (gauge, count of flagged pairs)

**Attack Detection Metrics:**
- `insignia_attack_breach` (gauge, per attack type, 0/1)
- `insignia_attack_severity` (gauge, per attack type, 0-1)
- `insignia_total_breaches` (gauge, count of active breaches)

**Optimizer Metrics:**
- `insignia_generation` (counter, current evolutionary generation)
- `insignia_best_fitness` (gauge, per objective)
- `insignia_population_diversity` (gauge)
- `insignia_pareto_front_size` (gauge)

**Implementation:** `tuning/metrics_exporter.py` — Prometheus client library exposing metrics on `:8000/metrics`.

#### Grafana Dashboard

Pre-configured dashboard with panels for:
1. **Scoring Distribution** — histogram of model/trading/pair composite scores
2. **Attack Status** — traffic light panel for each attack vector
3. **Weight Evolution** — how scoring weights change across generations
4. **Pareto Front** — scatter plot of multi-objective fitness
5. **P&L Distribution** — trader strategy returns across pairs
6. **Pairing Flow** — pairs per generation, Pareto front size, and collusion flags

**Implementation:** `monitoring/` directory with `docker-compose.yml`, Prometheus config, and Grafana provisioning.

---

### Phase 5: Evolutionary Multi-Objective Optimization

**Goal:** Automatically search the parameter space to maximize honest miner performance while minimizing attack vector breaches.

#### Optimization Objectives (minimize all)

1. **-mean_honest_score:** Negative mean composite score of honest miners (maximize honest performance)
2. **attack_breach_rate:** Fraction of attack vectors that are breached
3. **score_variance:** Variance of honest miner scores (prefer stability)
4. **-score_separation:** Negative gap between honest and adversarial miner scores (maximize separation)

#### Algorithm: NSGA-II (pymoo)

- **Population size:** 30
- **Generations:** 20 (configurable)
- **Crossover:** Simulated Binary Crossover (SBX)
- **Mutation:** Polynomial Mutation
- **Constraint handling:** model + trading weight sums = 1.0 (repair operator)

#### Current report-aligned operating note

The third orchestration run advances the system from "secure and improving" to
**target achieved** status:

- NSGA-II v13 R2 reached the target at knee point `V13-R2-KP-020-a7f2`
- breach_rate: `3.5e-6`
- honest_score: `0.9795`
- separation: `0.953`
- variance: `0.0009`
- target was hit by generation `12`, with the remaining generations refining a
  `21`-member Pareto front
- hypervolume increased from `0.0018` to `0.0161`
- Gaussian Process surrogate accuracy reached `R^2 = 0.93` on `93` empirical
  data points
- commit-reveal effectiveness strengthened to `0.76` in sentinel coordination
  and `0.801` in the simulator's pre/post stability validation

The earlier simulation-layer `0.124` / `0.847` benchmark is still valuable as a
stress environment, but the operating question has changed: the remaining work
is now empirical validation of the achieved `3.5e-6` breach rate and direct
measurement of PC-VH-006's production impact on Sybil severity.

#### Fitness Evaluation Pipeline

```
For each individual in population:
  1. Decode parameter vector → WeightConfig + PairingConfig + ...
  2. Inject parameters into simulation harness
  3. Run pair-based generations with all agent types (honest + adversarial)
  4. Collect per-miner quality + emission weights, pair fitnesses, breach reports
  5. Compute fitness vector: [honest_perf, breach_rate, variance, separation]
  6. Export metrics to Prometheus
  7. Return fitness to optimizer
```

**Implementation:** `tuning/optimizer.py` — pymoo Problem subclass + runner.

---

### Phase 6: Orchestrator

**Goal:** Tie everything together into a single entry point that runs the full automated tuning loop.

```bash
python -m tuning.orchestrator \
  --mode optimize \
  --generations 20 \
  --population 30 \
  --n-honest 6 \
  --n-adversarial 1 \
  --n-epochs 100 \
  --n-steps 150 \
  --output results/
```

The orchestrator:
1. Initializes the parameter space
2. Starts Prometheus metrics server
3. Initializes pymoo optimizer
4. For each generation:
   a. Evaluates all individuals via simulation harness
   b. Runs attack detection
   c. Computes fitness
   d. Exports metrics
   e. Saves checkpoints
5. Outputs the Pareto-optimal parameter configurations
6. Generates a summary report

**Implementation:** `tuning/orchestrator.py`

---

### Phase 7: Testnet Deployment Integration

**Goal:** After finding optimal parameters locally, deploy to Bittensor testnet for validation with real network dynamics.

#### Steps:
1. Export best parameter config as YAML
2. Inject into subnet code (WeightConfig, PairingConfig, etc.)
3. Deploy subnet on testnet: `btcli s create --subtensor.network test`
4. Run validator with optimized parameters
5. Deploy AI agent miners (same bots, but as real testnet neurons)
6. Monitor via Prometheus/Grafana connected to testnet validators
7. Compare testnet results to simulation predictions
8. Iterate if divergence > threshold

---

## Quick Start

```bash
# Install dependencies
cd subnet
pip install -r requirements.txt

# Run a single simulation with default parameters
python -m tuning.simulation

# Run attack detection
python -m tuning.attack_detector

# Run the full evolutionary optimization
python -m tuning.orchestrator --generations 20 --population 30

# Start monitoring stack (requires Docker)
cd monitoring
docker-compose up -d
# Grafana at http://localhost:3000 (admin/admin)
# Prometheus at http://localhost:9090
```

---

## Expected Outcomes

After running the optimizer:
- **Pareto front** of parameter configurations trading off honest performance vs. attack resistance
- **Breach report** showing which attack vectors are neutralized under each configuration
- **Recommended configuration** selected from Pareto front (knee point)
- **Confidence interval** from Monte Carlo reruns with different random seeds
- **Visualization** of how parameters evolved across generations
