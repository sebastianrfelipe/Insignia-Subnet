# Insignia Subnet — Parameter Tuning Strategy

## Executive Summary

The Insignia subnet's incentive mechanism involves **two composite scoring vectors** (L1: 6 metrics, L2: 6 metrics), **variance penalty hyperparameters**, **cross-layer feedback parameters**, **anti-gaming thresholds**, and **subnet hyperparameters**. Tuning these parameters by hand is infeasible because:

1. The parameter space is high-dimensional (33 tunable parameters)
2. Interactions between parameters are non-linear (e.g., changing the variance penalty lambda affects which models survive cross-regime evaluation, which determines what reaches L2)
3. Attack vectors exploit specific parameter configurations
4. You cannot iterate quickly on mainnet — miners are real actors with real stakes

**Solution:** Deploy on Bittensor testnet, replace real miners with AI agent bots that simulate honest, adversarial, and degenerate strategies, instrument everything with Prometheus/Grafana, and use evolutionary multi-objective optimization (pymoo NSGA-II) to search the parameter space automatically.

---

## L1 Scoring Design Philosophy

The L1 incentive mechanism is built on the **variance-penalized mean - lambda * std formulation** borrowed from robust GBDT hyperparameter optimization. Instead of evaluating a model on a single holdout window and computing point-estimate metrics, the validator:

1. Splits the out-of-sample benchmark data into **K rolling windows** covering diverse market regimes
2. Computes per-window F1 and Sharpe scores, producing a distribution
3. Reports the **penalized metric**: `mean(scores) - lambda * std(scores)`

This formulation simultaneously rewards:
- **High absolute performance** (the mean term)
- **Low variance / generalization across regimes** (the std penalty)

A separate **generalization gap = |train_f1 - val_f1|** directly measures overfitting without needing a complex detector — a model that memorizes training data will have a large gap; a model that generalizes will have a small gap.

### L1 Scoring Vector (6 dimensions)

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| `penalized_f1` | `mean_f1 - lambda * std_f1` | Directional prediction quality + cross-regime consistency |
| `penalized_sharpe` | `mean_sharpe - lambda * std_sharpe` | Risk-adjusted returns + cross-regime consistency |
| `max_drawdown` | worst peak-to-trough loss | Tail risk / fragility |
| `generalization_gap` | `\|train_f1 - val_f1\|` | Overfitting severity (minimize) |
| `feature_efficiency` | `1 / (1 + log(n_features / 10))` | Penalizes excessive feature count |
| `latency` | `exp(-(ms - target) / target)` | Inference speed for deployment viability |

### Why mean - lambda * std Over Separate Stability Metrics

The previous design used 7 metrics including a dedicated `stability_score` that computed cross-window coefficient of variation separately from accuracy and Sharpe. This had two problems:

1. **Redundancy:** Stability was being measured independently from the metric it was stabilizing. A model could game a high accuracy + high stability by being consistently mediocre.
2. **Weight interference:** The optimizer had to balance 7 weights where two of them (accuracy + stability) were trying to capture the same underlying quality.

The mean - lambda * std formulation collapses both concerns into a single metric per dimension. A model that scores 0.8 F1 with 0.02 std is strictly better than a model that scores 0.85 F1 with 0.15 std (at lambda = 0.5). The tradeoff is explicit in a single number.

### Why generalization_gap Over OverfittingDetector

The previous design used a `ReferenceOverfittingDetector` with gap thresholds, exponential decay rates, and model-complexity-weighted penalties. This introduced 2 internal hyperparameters (`gap_threshold`, `decay_rate`) that interacted non-linearly with the composite weight on overfitting.

The direct `|train_f1 - val_f1|` formulation:
- Has zero internal hyperparameters — only the composite weight on `generalization_gap` matters
- Is immediately interpretable (a gap of 0.15 means the model is 15% worse out-of-sample)
- Maps directly to the GBDT HPO objective that has been validated in production quantitative modeling

### Why F1 Over Raw Directional Accuracy

Financial prediction datasets are inherently class-imbalanced — markets trend more than they mean-revert, or vice versa depending on the regime. Raw accuracy rewards a model that predicts the majority class. F1 score (harmonic mean of precision and recall) forces the model to perform well on both up and down predictions, which is critical for a trading strategy that needs to go both long and short.

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

| Category | Parameters | Count |
|----------|-----------|-------|
| L1 Scoring Weights | penalized_f1, penalized_sharpe, max_drawdown, generalization_gap, feature_efficiency, latency | 6 |
| L2 Scoring Weights | realized_pnl, omega, max_drawdown, win_rate, consistency, model_attribution | 6 |
| Variance Penalty | lambda (variance_penalty), K (n_eval_windows) | 2 |
| Cross-Layer Promotion | top_n, min_consecutive_epochs, max_overfitting_score, max_score_decay_pct, expiry_epochs_without_usage | 5 |
| Cross-Layer Feedback | feedback_bonus_weight, feedback_penalty_weight | 2 |
| Anti-Gaming | correlation_threshold (fingerprint), copy_trade_time_tolerance, copy_trade_size_tolerance, copy_trade_correlation_threshold | 4 |
| Buyback | buyback_pct, min_profit_threshold | 2 |
| L2 Trading | base_spread_bps, volatility_impact_factor, size_impact_factor, fee_bps, max_position_pct, max_drawdown_pct | 6 |
| **Total** | | **33** |

**Key Constraints:**
- L1 weights must sum to 1.0
- L2 weights must sum to 1.0
- All weights in [0.01, 0.45]
- lambda (variance_penalty) in [0.1, 1.5] — controls how aggressively variance is punished
- K (n_eval_windows) in [3, 10] — number of rolling windows for mean/std computation

**Implementation:** `tuning/parameter_space.py` — defines the search space, encoding/decoding, weight-sum repair operator, and constraint handling.

---

### Phase 2: Simulation Harness with AI Agent Miners

**Goal:** Build a self-contained simulation that runs the full subnet pipeline with configurable bot miners instead of real network participants.

#### Miner Agent Types

| Agent Type | Behavior | Purpose |
|------------|----------|---------|
| **Honest** | Trains models normally, submits best effort | Baseline — represents good-faith miners |
| **Overfitter** | High complexity, low regularization to maximize train-val gap | Tests generalization_gap detection |
| **Copycat** | Copies another miner's model with small perturbations | Tests plagiarism detection |
| **Single-Metric Gamer** | Optimizes only for F1 at the expense of Sharpe and drawdown | Tests composite scoring robustness |
| **Sybil** | Multiple identities with correlated models from same seed | Tests sybil detection via fingerprinting |
| **Random** | Random model/strategy submissions | Noise floor — tests that scoring discriminates signal from noise |
| **Honest Trader (L2)** | Standard paper trading using promoted L1 models | L2 baseline |
| **Copy-Trader (L2)** | Mirrors another L2 miner's positions | Tests copy-trade detection |

Each agent type is parameterized so the optimizer can test different attack intensities.

**Implementation:** `tuning/simulation.py` — contains `MinerAgent` classes, `SimulationHarness`, and the full pipeline runner.

---

### Phase 3: Attack Vector Detection

**Goal:** Automatically detect whether each of the 9 documented attack vectors has been breached under a given parameter configuration.

For each attack vector, define a **breach condition** (boolean) and a **severity score** (0.0 = fully defended, 1.0 = fully exploited):

| Attack | Breach Condition |
|--------|-----------------|
| Overfitting exploitation | Overfitting miner's mean score > honest miners' mean score (generalization_gap weight too low or lambda too permissive) |
| Model plagiarism (L1) | Copycat miner evades fingerprinting and scores above 80% of honest mean |
| Single-metric gaming | Single-metric gamer (F1-only optimizer) ranks in top 50% — means composite weights are poorly balanced |
| Sybil attack | Sybil cluster collectively earns >1.5x their fair share |
| Copy-trading (L2) | Copy-trader evades detection and scores above 70% of honest L2 mean |
| Random baseline discrimination | Random miner scores >60% of honest mean — scoring mechanism doesn't discriminate |
| Adversarial dominance | An adversarial miner is the #1 ranked miner overall |
| Insufficient separation | Gap between honest and adversarial mean scores is less than 10% |
| Score concentration | Herfindahl index of emission weights exceeds 3x the fair-share baseline |

**Implementation:** `tuning/attack_detector.py` — runs the simulation with adversarial agents and returns a breach report with per-attack severity.

---

### Phase 4: Prometheus Metrics & Grafana Dashboards

**Goal:** Instrument the simulation with metrics so you can visualize tuning progress in real-time.

#### Metrics Exported

**Simulation Metrics:**
- `insignia_l1_composite_score` (gauge, per miner, per agent type)
- `insignia_l1_consensus_weight` (gauge, per miner)
- `insignia_l2_composite_score` (gauge, per strategy, per agent type)
- `insignia_l2_realized_pnl` (gauge, per strategy)
- `insignia_l2_max_drawdown` (gauge, per strategy)
- `insignia_promotion_active_count` (gauge, active models in L2 pool)
- `insignia_feedback_adjustment` (gauge, per model)

**Attack Detection Metrics:**
- `insignia_attack_breach` (gauge, per attack type, 0/1)
- `insignia_attack_severity` (gauge, per attack type, 0-1)
- `insignia_total_breaches` (gauge, count of active breaches)

**Optimizer Metrics:**
- `insignia_optimizer_generation` (counter, current evolutionary generation)
- `insignia_best_fitness` (gauge, per objective)
- `insignia_population_diversity` (gauge)
- `insignia_pareto_front_size` (gauge)

**Implementation:** `tuning/metrics_exporter.py` — lightweight Prometheus-compatible HTTP server exposing metrics on `:8000/metrics`.

#### Grafana Dashboard

Pre-configured dashboard with panels for:
1. **L1/L2 Scoring Distribution** — per-miner composite scores colored by agent type
2. **Attack Status** — traffic light panel for each attack vector (green/red)
3. **Attack Severity** — bar gauge showing per-attack severity 0-1
4. **Optimizer Progress** — generation counter, Pareto front size, population diversity
5. **Best Fitness per Objective** — bar gauge of the 4 optimization objectives
6. **Promotion Pool Size** — time series of active models promoted to L2
7. **Cross-Layer Feedback** — time series of feedback adjustment multipliers per model

**Implementation:** `monitoring/` directory with `docker-compose.yml`, Prometheus config, and Grafana provisioning with pre-built dashboard JSON.

---

### Phase 5: Evolutionary Multi-Objective Optimization

**Goal:** Automatically search the 33-dimensional parameter space to maximize honest miner performance while minimizing attack vector breaches.

#### Optimization Objectives (all minimized)

| Objective | Formula | What It Drives |
|-----------|---------|---------------|
| `neg_honest_score` | `-mean(honest_composite_scores)` | Maximize honest miner performance |
| `breach_rate` | `n_breached / n_total_attacks` | Minimize attack vector exposure |
| `score_variance` | `var(honest_composite_scores)` | Prefer consistent scoring across honest miners |
| `neg_separation` | `-(mean_honest - mean_adversarial)` | Maximize gap between honest and adversarial miners |

#### Algorithm: NSGA-II (pymoo)

- **Population size:** 30-50 (configurable)
- **Generations:** 50-100 (configurable)
- **Crossover:** Simulated Binary Crossover (SBX), eta=15
- **Mutation:** Polynomial Mutation, eta=20
- **Constraint handling:** Custom repair operator normalizes L1/L2 weight groups to sum to 1.0 after crossover/mutation
- **Fallback:** Random search with scalarized objective when pymoo is unavailable

#### Fitness Evaluation Pipeline

```
For each individual in population:
  1. Decode 33-dim parameter vector → WeightConfig (with λ, K) + PromotionConfig + ...
  2. Inject parameters into simulation harness
  3. Run full pipeline: L1 agents → L1 Validator → Promotion → L2 agents → L2 Validator → Feedback
  4. Run attack detection on simulation results
  5. Compute 4-objective fitness vector: [neg_honest, breach_rate, variance, neg_separation]
  6. Export metrics to Prometheus
  7. Return fitness to NSGA-II
```

**Selection pressure on key parameters:**
- **lambda (variance_penalty):** Higher lambda more aggressively penalizes models with inconsistent cross-regime performance. Too high filters out all models; too low lets overfitters through.
- **generalization_gap weight:** Higher weight makes overfitting the dominant failure mode. Interacts with lambda — both need to be tuned together.
- **Anti-gaming thresholds:** Too strict causes false positives on honest miners; too lenient lets adversaries through.

**Implementation:** `tuning/optimizer.py` — pymoo Problem subclass + NSGA-II runner + random search fallback.

---

### Phase 6: Orchestrator

**Goal:** Tie everything together into a single entry point that runs the full automated tuning loop.

```bash
# Quick sanity check with default parameters
python -m tuning.orchestrator --mode single

# Statistical attack analysis across multiple trials
python -m tuning.orchestrator --mode attack --trials 10

# Full evolutionary optimization
python -m tuning.orchestrator --mode optimize \
  --generations 50 \
  --population 30 \
  --n-honest 6 \
  --n-adversarial 1 \
  --n-epochs 2 \
  --output results/

# Random search (no pymoo required)
python -m tuning.orchestrator --mode random --population 30
```

The orchestrator:
1. Initializes the 33-parameter search space
2. Starts Prometheus metrics server on `:8000/metrics`
3. Initializes pymoo NSGA-II optimizer (or random search fallback)
4. For each generation:
   a. Evaluates all individuals via simulation harness
   b. Runs attack detection on each simulation result
   c. Computes 4-objective fitness vector
   d. Exports metrics to Prometheus
   e. Saves checkpoints
5. Selects the knee point from the Pareto front
6. Exports the best configuration as YAML for subnet deployment
7. Generates a summary report

**Implementation:** `tuning/orchestrator.py`

---

### Phase 7: Testnet Deployment Integration

**Goal:** After finding optimal parameters locally, deploy to Bittensor testnet for validation with real network dynamics.

#### Steps:
1. Export best parameter config as YAML (`results/best_config.yaml`)
2. Inject into subnet code — the YAML maps directly to `WeightConfig` (including lambda and K), `PromotionConfig`, feedback weights, anti-gaming thresholds, and trading engine parameters
3. Deploy subnet on testnet: `btcli s create --subtensor.network test`
4. Run validator with optimized parameters
5. Deploy AI agent miners (same bot implementations, but running as real testnet neurons)
6. Monitor via Prometheus/Grafana connected to testnet validators
7. Compare testnet results to simulation predictions — specifically:
   - Do honest miners still score highest?
   - Is the generalization gap distribution similar?
   - Are the same attack vectors defended?
8. Iterate if divergence > threshold

---

## Quick Start

```bash
# Install dependencies
cd subnet
pip install -r requirements.txt

# Run a single simulation with default parameters
python3 -m tuning.simulation

# Run attack detection
python3 -m tuning.attack_detector

# Run the full orchestrator in single-simulation mode
python3 -m tuning.orchestrator --mode single

# Run evolutionary optimization
python3 -m tuning.orchestrator --mode optimize --generations 20 --population 30

# Start monitoring stack (requires Docker)
cd monitoring
docker-compose up -d
# Grafana at http://localhost:3000 (admin/admin)
# Prometheus at http://localhost:9090
```

---

## Expected Outcomes

After running the optimizer:
- **Pareto front** of parameter configurations trading off honest performance vs. attack resistance vs. score variance vs. honest-adversarial separation
- **Breach report** showing which attack vectors are neutralized under each configuration
- **Recommended configuration** selected from Pareto front (knee point), exported as YAML
- **Optimal lambda and K** — the variance penalty coefficient and number of rolling windows that best balance performance vs. generalization
- **Confidence interval** from Monte Carlo reruns with different random seeds
- **Visualization** in Grafana of how parameters evolved across generations
