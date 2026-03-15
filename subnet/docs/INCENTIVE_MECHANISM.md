# Insignia Subnet — Incentive Mechanism Design

## Overview

The Insignia incentive mechanism ensures that miners are rewarded proportionally to the genuine quality and deployability of their contributions, while making all known gaming strategies unprofitable.

The design operates across two independent layers, each with its own Yuma consensus cycle, connected by a cross-layer feedback loop.

---

## Layer 1: Model Generation Incentives

### Scoring Vector (6 Dimensions)

Metrics are computed across K rolling out-of-sample windows. The core innovation is the **variance-penalized mean - lambda * std formulation**: each metric rewards both high absolute performance (mean) and low variance across market regimes (std penalty).

| Metric | Default Weight | Formula | Purpose |
|--------|---------------|---------|---------|
| Penalized F1 | 25% | `mean_f1 - lambda * std_f1` | Directional quality + cross-regime consistency in one metric |
| Penalized Sharpe | 25% | `mean_sharpe - lambda * std_sharpe` | Risk-adjusted returns + cross-regime consistency |
| Max Drawdown | 15% | worst peak-to-trough | Penalizes fragile models with tail risk |
| Generalization Gap | 20% | `\|train_f1 - val_f1\|` | Direct overfitting measure — lower is better |
| Feature Efficiency | 5% | `1/(1+log(n/10))` | Penalizes models requiring excessive features |
| Latency | 10% | `exp(-(ms-target)/target)` | Inference speed — critical for short-horizon deployment |

### Why This Drives Good Behavior

- **Variance-penalized scoring** prevents models that spike in one regime and collapse in another. The mean - lambda * std formulation bakes consistency into every metric rather than measuring it as a separate dimension.
- **Generalization gap** directly targets the most common failure mode of GBDTs on financial data: a model with 0.85 train F1 and 0.60 val F1 has a gap of 0.25, which is heavily penalized.
- **F1 over accuracy** handles class imbalance in financial predictions — a model must perform well on both up and down predictions.
- **Feature efficiency** discourages models that depend on data sources that won't be available in production.

### Emission Distribution

Top-performing L1 miners earn alpha token emissions via Yuma consensus. Weights are proportional to composite scores. The top-N models are also promoted to the Layer 2 pool, creating a second revenue stream.

---

## Layer 2: Strategy Deployment Incentives

### Scoring Vector (6 Dimensions)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Realized P&L | 25% | Absolute returns from actual trading outcomes |
| Omega Ratio | 20% | Full-distribution risk measure (captures tail behavior) |
| Max Drawdown | 15% | Hard ceiling — breach eliminates the strategy entirely |
| Win Rate | 10% | Signal precision — penalizes low-conviction noise trading |
| Consistency | 20% | Rolling 7-day sub-window analysis — penalizes spike-then-collapse |
| Model Attribution | 10% | Credit to miners using models with strong L2 track records |

### Why This Drives Good Behavior

- **Real outcomes only**: L2 scores are based on actual (paper/live) trading results, not simulations.
- **Drawdown elimination**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated, mirroring institutional prop trading standards.
- **Consistency requirements** prevent strategies that take one lucky trade and coast.
- **Model attribution** creates incentive alignment between L1 and L2 miners.

---

## Cross-Layer Feedback Loop

The two layers create a self-reinforcing quality signal:

```
L1 Models ──> Promotion ──> L2 Strategies ──> L2 Scores ──┐
    ↑                                                       │
    └──── Retroactive Bonus/Penalty ◄──────────────────────┘
```

1. **Bonus**: L1 models whose L2 strategies score > 0.6 receive a retroactive multiplier (up to +15%) in the next L1 epoch.
2. **Penalty**: L1 models whose L2 strategies score < 0.3 receive a penalty (up to -10%) despite good simulation scores.
3. **Minimum evidence**: Adjustments only apply after 3+ L2 epochs of data to prevent noise.

This closes the simulation-to-reality gap: models are ultimately judged by deployment outcomes.

---

## Attack Vector Analysis

### 1. Overfitting to Public Data

| | |
|---|---|
| **Attack** | Miner memorizes patterns in publicly available data that correlate with the validation window. |
| **Defense** | Validators score against proprietary tick-by-tick data miners cannot access. Rolling holdout windows change each epoch. The generalization_gap metric (\|train_f1 - val_f1\|) directly penalizes models with in-sample/out-of-sample divergence. |
| **Why it fails** | The data asymmetry is the core moat. Models that only memorize public patterns will fail the proprietary benchmark. The generalization gap catches overfitting regardless of how sophisticated the memorization is. |

### 2. Submission Spam / Brute Force

| | |
|---|---|
| **Attack** | Miner submits many model variants per epoch to maximize chance of a lucky high score. |
| **Defense** | Rate limited to 1 submission per miner per epoch (24h minimum). Each submission requires a full metadata manifest. |
| **Why it fails** | At 1 submission/day, brute-forcing is impractical. Each attempt costs compute time. |

### 3. Model Plagiarism (L1)

| | |
|---|---|
| **Attack** | Miner copies another miner's model artifact or reverse-engineers their approach. |
| **Defense** | SHA-256 fingerprinting detects exact duplicates. Prediction correlation analysis detects behavioral clones. Correlated models share rewards. |
| **Why it fails** | No incentive to copy — you only get a fraction of the reward. Original work pays more. |

### 4. Copy-Trading (L2)

| | |
|---|---|
| **Attack** | L2 miner mirrors another miner's positions instead of building their own strategy. |
| **Defense** | Position correlation analysis with time/size tolerance. Correlated strategies share rewards. |
| **Why it fails** | Same as model plagiarism — copying dilutes your reward. |

### 5. Single-Metric Gaming

| | |
|---|---|
| **Attack** | Miner optimizes for one dominant metric while ignoring others. |
| **Defense** | Composite scoring across 6 L1 / 6 L2 metrics. No single metric dominates (max weight 25%). The variance-penalized mean - lambda*std formulation means gaming F1 alone fails if Sharpe or drawdown is poor. |
| **Why it fails** | High F1 with high drawdown scores poorly. High Sharpe with large generalization gap scores poorly. The mean-std formulation further penalizes models that only perform well in specific regimes. |

### 6. Validator Data Leakage

| | |
|---|---|
| **Attack** | Miner reverse-engineers the proprietary validation dataset from score feedback. |
| **Defense** | Only aggregate scores returned, never raw predictions or data. Rolling windows change each epoch. Historical windows released 30 days after evaluation. |
| **Why it fails** | Aggregate scores reveal almost no information about the underlying data distribution. |

### 7. L2 Paper Trading Manipulation

| | |
|---|---|
| **Attack** | L2 miner fabricates paper trading results or cherry-picks favorable reporting windows. |
| **Defense** | Validators track positions via continuous streaming. All positions are timestamped. Reporting gaps are penalized. |
| **Why it fails** | Validators independently verify position state — fabricated results are immediately detected. |

### 8. Sybil Attack

| | |
|---|---|
| **Attack** | Single entity registers multiple miner identities to capture more emissions. |
| **Defense** | Model fingerprinting + prediction correlation catch behavioral duplicates. Bittensor's staking requirements raise the cost of sybil identities. |
| **Why it fails** | Each identity must submit genuinely different, high-quality work to earn rewards. The cost of staking N identities scales linearly while rewards per identity decrease. |

### 9. Regime-Specific Exploitation

| | |
|---|---|
| **Attack** | Model only works in specific market conditions (e.g., bull market) and fails in others. |
| **Defense** | The variance-penalized scoring (mean_f1 - lambda*std_f1 and mean_sharpe - lambda*std_sharpe) directly penalizes models with high cross-regime variance. Validation windows deliberately cover trending, ranging, high-vol, low-vol, and crisis periods. |
| **Why it fails** | A model that scores 0.85 F1 in bull markets but 0.45 in bear markets has a high std, which is subtracted from the mean. Consistent 0.70 across all regimes scores higher. |

---

## Buyback Mechanism

Profits from deployed model+strategy pairs create a virtuous economic cycle:

```
Better Models → Higher Firm P&L → Token Buybacks → Higher Token Value
       ↑                                                      │
       └──────── Stronger Miner Incentive ◄───────────────────┘
```

- **Buyback percentage**: 20% of firm deployment P&L (configurable)
- **Minimum threshold**: Buybacks trigger only above a minimum P&L floor
- **Frequency**: Weekly buyback cycles
- **Transparency**: Buyback amounts and timing are published on-chain

This creates a direct link between the subnet's economic output and miner token value — an alignment mechanism that existing trading subnets lack.

---

## Economic Sustainability

| Revenue Stream | Description |
|---|---|
| **Internal Deployment** (Primary) | Subnet owner's firm deploys winning pairs in live prop trading. Direct P&L justifies infrastructure costs. |
| **Emissions Bootstrap** | Bittensor network emissions fund miner participation before external revenue scales. |
| **Buyback Loop** | Deployment profits → token buybacks → higher miner incentives → better models. |
| **External Signal API** (Future) | Package L2 strategy outputs as subscription service for external quant funds. |
| **Model Marketplace** (Future) | License winning model architectures to external ML teams. |

The subnet owner being the primary consumer of L2 output creates a demand floor that does not depend on external market adoption.
