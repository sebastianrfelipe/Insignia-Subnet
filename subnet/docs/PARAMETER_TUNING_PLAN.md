# Insignia Subnet — Parameter Tuning Strategy

## Executive Summary

The Insignia subnet's incentive mechanism involves **two composite scoring vectors** (L1: 7 metrics, L2: 10 metrics), **cross-layer feedback parameters**, **anti-gaming thresholds**, **commit-reveal parameters**, **emission distribution parameters**, and **subnet hyperparameters**. The full tuning landscape spans two distinct levels:

- **55 Insignia application-level parameters** — scoring weights, thresholds, commit-reveal timing, defense parameters, emission distribution, and mechanism knobs (tuned by the emulator)
- **39 Bittensor on-chain subnet hyperparameters** — network-level parameters controlling registration, consensus, staking, and bonds (set via `btcli subnets hyperparameters`)

Together these form a **94-parameter optimization surface**. Tuning by hand is infeasible because:

1. The parameter space is high-dimensional (80 total parameters across two levels)
2. Interactions between parameters are non-linear (e.g., changing L1 overfitting weight affects which models reach L2; changing `tempo` affects how frequently weights are set)
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
       │   ┌─────────────────┐        │               │
       │   │ Subnet Pipeline │        │               │
       │   │ L1 Miners (bots)│        │               │
       │   │ L1 Validator    │◄───────┘               │
       │   │ L2 Miners (bots)│                        │
       │   │ L2 Validator    │                        │
       │   │ Cross-Layer     │                        │
       │   └────────┬────────┘                        │
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

#### Insignia Application-Level Parameters (41 total, tuned by emulator)

| Category | Parameters | Count |
|----------|-----------|-------|
| L1 Scoring Weights | penalized_f1, penalized_sharpe, max_drawdown, variance_score, overfitting_penalty, feature_efficiency, latency | 7 |
| L2 Scoring Weights | realized_pnl, omega, max_drawdown, win_rate, consistency, model_attribution, execution_quality, annualized_volatility, sharpe_ratio, sortino_ratio | 10 |
| Overfitting Detector | gap_threshold, decay_rate | 2 |
| Cross-Layer Promotion | top_n, min_consecutive_epochs, max_overfitting_score, max_score_decay_pct, expiry_epochs | 5 |
| Cross-Layer Feedback | feedback_bonus_weight, feedback_penalty_weight | 2 |
| Anti-Gaming | fingerprint_correlation_threshold, copy_trade_time_tolerance, copy_trade_size_tolerance, copy_trade_correlation_threshold | 4 |
| L2 Trading | base_spread_bps, volatility_impact_factor, size_impact_factor, fee_bps, max_position_pct, max_drawdown_pct | 6 |
| Buyback | buyback_pct, min_profit_threshold | 2 |
| Emission Distribution | sigmoid_midpoint, sigmoid_steepness, l1_l2_emission_split | 3 |
| Rate Limiting | rate_limit_epoch_seconds | 1 |
| Feedback Thresholds | feedback_min_l2_epochs, feedback_bonus_threshold, feedback_penalty_threshold | 3 |
| Commit-Reveal Timing | commit_window_seconds, reveal_window_seconds, late_reveal_penalty | 3 |
| Validation Timing | min_prediction_lead_time, validator_latency_penalty_weight, high_latency_threshold_ms | 3 |
| Consensus Integrity | weight_entropy_minimum, cross_validator_score_variance_max, validator_rotation_max_consecutive_epochs, validator_agreement_threshold, collusion_detection_lookback_epochs | 5 |
| Cross-Layer Defense | cross_layer_penalty_strength, cross_layer_latency | 2 |
| **Subtotal** | | **55** |

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

| Level | Parameters | Tuned By |
|-------|-----------|----------|
| Insignia application-level | 55 | Emulator (NSGA-II evolutionary optimization) |
| Bittensor on-chain | 33+ | btcli / subnet owner configuration |
| **Total** | **88+** | |

**Constraints:**
- L1 weights must sum to 1.0
- L2 weights must sum to 1.0
- All weights ∈ [0.01, 0.50]
- Thresholds must be positive
- `feedback_penalty_threshold` < `feedback_bonus_threshold`

**Implementation:** `tuning/parameter_space.py` — defines the search space, encoding/decoding, and constraint handling.

---

### Phase 2: Simulation Harness with AI Agent Miners

**Goal:** Build a self-contained simulation that runs the full subnet pipeline with configurable bot miners instead of real network participants.

#### Miner Agent Types

| Agent Type | Behavior | Purpose |
|------------|----------|---------|
| **Honest** | Trains models normally, submits best effort | Baseline — represents good-faith miners |
| **Overfitter** | Deliberately overfits to training data | Tests overfitting detection |
| **Copycat** | Copies another miner's model with small perturbations | Tests plagiarism detection |
| **Spammer** | Submits many low-quality models rapidly | Tests rate limiting |
| **Single-Metric Gamer** | Optimizes only for directional accuracy | Tests composite scoring robustness |
| **Sybil** | Multiple identities submitting correlated models | Tests sybil detection |
| **Copy-Trader (L2)** | Mirrors another L2 miner's positions | Tests copy-trade detection |
| **Random** | Random model/strategy submissions | Noise floor baseline |

Each agent type is parameterized so the optimizer can test different attack intensities.

**Implementation:** `tuning/simulation.py` — contains `MinerAgent` classes, `SimulationHarness`, and the full pipeline runner.

---

### Phase 3: Attack Vector Detection

**Goal:** Automatically detect whether each of the 19 documented attack vectors has been breached under a given parameter configuration.

For each attack vector, define a **breach condition** (boolean) and a **severity score** (0-1):

| # | Attack | Breach Condition |
|---|--------|-----------------|
| 1 | Overfitting exploitation | Overfitting miner scores higher than honest miners |
| 2 | Submission spam | Spammer gets >0 score despite rate limits |
| 3 | Model plagiarism (L1) | Copycat miner is not detected and scores independently |
| 4 | Copy-trading (L2) | Copy-trader is not detected and scores independently |
| 5 | Single-metric gaming | Single-metric gamer ranks in top 50% |
| 6 | Validator data leakage | (Structural check — no simulation needed) |
| 7 | Paper trading manipulation | Inflated P&L not caught by slippage model |
| 8 | Sybil attack | Sybil identities collectively earn >2x single miner share |
| 9 | Regime-specific exploitation | Model that only works in 1 regime ranks in top 25% |
| 10 | L1/L2 weight skew exploitation | Adversarial miner captures disproportionate rewards via emission split |
| 11 | Cross-layer timing sync | Timing gaps between L1/L2 scoring windows exploited for feedback gaming |
| 12 | Objective weight manipulation | NSGA-II objective weights manipulated to favor specific attack patterns |
| 13 | GA parameter exploitation | Genetic algorithm parameters (crossover, mutation) tuned to exploit optimizer |
| 14 | Governance parameter manipulation | On-chain hyperparameters manipulated to enable other attacks |
| 15 | L1/L2 incentive misalignment | Emission split and weight ratios create cross-layer gaming opportunities |
| 16 | Pareto front manipulation | Population diversity or elite preservation exploited to bias optimization |
| 17 | Reward distribution manipulation | Sigmoid emission curve or buyback parameters gamed for disproportionate rewards |
| 18 | Validator latency exploitation (Vector 8) | Miner accuracy correlated with validator latency; post-market submissions scored as predictions. **Commit-reveal validated: projected severity 0.047 < 0.05 target** |
| 19 | Miner-validator collusion | Validator weight/score bias toward specific miners not justified by performance |

**Implementation:** `tuning/attack_detector.py` — runs the simulation with adversarial agents and returns a breach report. Currently at v4.0 with 19 vectors and full detection methods.

**Commit-reveal impact on Vector 18:** Sentinel validation (session 69dab601) confirmed that the commit-reveal scheme reduces Vector 18 severity from 0.09 to projected 0.047, clearing the 0.05 target. Companion Vector 11 (prediction timing manipulation) also drops from 0.06 to 0.025. The critical effectiveness threshold is 0.667 — this must be validated in simulation before production deployment.

---

### Phase 4: Prometheus Metrics & Grafana Dashboards

**Goal:** Instrument the simulation with metrics so you can visualize tuning progress in real-time.

#### Metrics Exported

**Simulation Metrics:**
- `insignia_l1_composite_score` (gauge, per miner, per epoch)
- `insignia_l1_weight` (gauge, per miner)
- `insignia_l2_composite_score` (gauge, per strategy)
- `insignia_l2_pnl` (gauge, per strategy)
- `insignia_l2_drawdown` (gauge, per strategy)
- `insignia_promotion_count` (counter)
- `insignia_feedback_adjustment` (gauge, per model)

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
1. **Scoring Distribution** — histogram of L1/L2 composite scores across miners
2. **Attack Status** — traffic light panel for each attack vector
3. **Weight Evolution** — how scoring weights change across generations
4. **Pareto Front** — scatter plot of multi-objective fitness
5. **P&L Distribution** — L2 strategy returns across miners
6. **Cross-Layer Flow** — promotion counts and feedback adjustments

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

- **Population size:** 50
- **Generations:** 100 (configurable)
- **Crossover:** Simulated Binary Crossover (SBX)
- **Mutation:** Polynomial Mutation
- **Constraint handling:** L1/L2 weight sum = 1.0 (repair operator)

#### Fitness Evaluation Pipeline

```
For each individual in population:
  1. Decode parameter vector → WeightConfig + PromotionConfig + ...
  2. Inject parameters into simulation harness
  3. Run simulation with all agent types (honest + adversarial)
  4. Collect L1/L2 scores, rankings, breach reports
  5. Compute fitness vector: [honest_perf, breach_rate, variance, separation]
  6. Export metrics to Prometheus
  7. Return fitness to optimizer
```

**Implementation:** `tuning/optimizer.py` — pymoo Problem subclass + runner.

---

### Phase 6: Orchestrator

**Goal:** Tie everything together into a single entry point that runs the full automated tuning loop.

```bash
python tuning/orchestrator.py \
  --generations 100 \
  --population 50 \
  --n-honest-miners 10 \
  --n-adversarial-miners 6 \
  --n-epochs 3 \
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
2. Inject into subnet code (WeightConfig, PromotionConfig, etc.)
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
