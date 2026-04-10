# Insignia Subnet — Incentive Mechanism Design

## Overview

The Insignia incentive mechanism ensures that miners are rewarded proportionally to the genuine quality and deployability of their contributions, while making all known gaming strategies unprofitable.

The design operates across two independent layers, each with its own Yuma consensus cycle, connected by a cross-layer feedback loop.

---

## Layer 1: Model Generation Incentives

### Scoring Vector (7 Dimensions)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Penalized F1 | 22% | Directional prediction quality with cross-regime consistency penalty (mean − λ·std across windows) |
| Penalized Sharpe Ratio | 18% | Risk-adjusted returns with variance penalty across rolling sub-windows |
| Max Drawdown | 14% | Penalizes fragile models with large peak-to-trough losses |
| Variance Score | 16% | Cross-regime consistency — measures coefficient of variation across market regimes |
| Overfitting Penalty | 14% | Gap between in-sample and out-of-sample performance (proprietary metric) |
| Feature Efficiency | 6% | Penalizes models requiring exotic or excessive features |
| Latency Score | 10% | Inference speed — critical for short-horizon deployment |

All metrics use a **variance-penalized formulation** (`mean − λ·std`) across rolling windows, rewarding both peak performance and consistency.

### Why This Drives Good Behavior

- **Multi-dimensional scoring** prevents miners from gaming a single metric. A model with high F1 but 40% max drawdown scores poorly.
- **Variance-penalized metrics** ensure that high aggregate scores cannot be achieved through a single lucky window while being inconsistent elsewhere.
- **Overfitting detection** specifically targets the most common failure mode of GBDTs on financial data.
- **Variance Score** ensures models work across market regimes, not just the current one.
- **Feature efficiency** discourages models that depend on data sources that won't be available in production.

### Emission Distribution

Top-performing L1 miners earn alpha token emissions via Yuma consensus. Weights are proportional to composite scores. The top-N models are also promoted to the Layer 2 pool, creating a second revenue stream.

---

## Layer 2: Strategy Deployment Incentives

### Scoring Vector (10 Dimensions)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Realized P&L | 17% | Absolute returns from actual trading outcomes |
| Omega Ratio | 13% | Full-distribution risk measure (captures tail behavior) |
| Max Drawdown | 13% | Hard ceiling — breach eliminates the strategy entirely |
| Win Rate | 8% | Signal precision — penalizes low-conviction noise trading |
| Consistency | 13% | Rolling 7-day sub-window analysis — penalizes spike-then-collapse |
| Model Attribution | 8% | Credit to miners using models with strong L2 track records |
| Execution Quality | 13% | Latency, reliability, and slippage — infrastructure health |
| Annualized Volatility | 5% | Cumulative realized volatility — lower = better score |
| Sharpe Ratio | 5% | Risk-adjusted return per unit of total volatility |
| Sortino Ratio | 5% | Risk-adjusted return per unit of downside volatility |

### Metric Definitions

#### 1. Realized P&L (20%)

Measures the strategy's raw profitability relative to a baseline (typically buy-and-hold or zero). This is the most direct measure of whether a strategy generates economic value.

```
score = clamp((pnl - baseline) / max(|baseline|, 1.0), 0, 1)
```

- Strategies at or below the baseline receive a score of zero.
- The denominator scales by the absolute baseline value so that the metric is meaningful across different capital levels and market conditions.
- Carries the highest single L2 weight (17%) because realized returns are the ultimate objective of the subnet.

**Normalization**: Already in [0, 1] from the scoring function itself.

#### 2. Omega Ratio (15%)

A full-distribution risk measure that captures the complete shape of the return distribution, including skewness and fat tails. Unlike Sharpe ratio (which only considers mean and variance), Omega reflects the probability-weighted balance between gains and losses at a given threshold.

```
Omega = sum(max(r_i - threshold, 0)) / sum(max(threshold - r_i, 0))
```

- An Omega > 1 means the strategy's gain mass exceeds its loss mass at the threshold.
- Raw values are capped at 10.0 to prevent degenerate cases (e.g., a single winning trade with no losses) from dominating.
- This metric is critical for crypto markets, where returns are rarely normally distributed and tail risk is the primary destroyer of capital.

**Normalization**: Divided by 3.0, so Omega >= 3.0 maps to a perfect normalized score of 1.0. This threshold reflects that an Omega of 3+ is exceptional for crypto trading strategies.

#### 3. Max Drawdown (15%)

The peak-to-trough loss of the strategy's equity curve. This metric has a unique dual role:

1. **Soft score component**: Lower drawdown = higher normalized score (`1 - drawdown`).
2. **Hard elimination threshold**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated and receive a composite score of zero for the epoch, mirroring institutional prop trading standards.

```
drawdown = (peak_equity - current_equity) / peak_equity
normalized = 1.0 - drawdown
```

**Normalization**: Inverted so that lower drawdown = higher score. A 0% drawdown yields 1.0; a 100% drawdown yields 0.0.

#### 4. Win Rate (10%)

The fraction of trades that were profitable. A straightforward measure of signal precision.

```
win_rate = count(trade_pnl > 0) / total_trades
```

- Carries the lowest weight (10%) by design, because profitable strategies can legitimately have moderate win rates (e.g., trend-following with ~40% wins but large risk/reward ratios).
- Its primary role is to filter out noise trading: strategies that generate excessive churn without directional edge.

**Normalization**: Already in [0, 1] from the scoring function.

#### 5. Consistency (15%)

Rolling sub-window analysis that penalizes "spike-then-collapse" strategies. This is the strongest predictor of a strategy's viability in live deployment.

The metric divides the return history into non-overlapping 7-day windows and computes a Sharpe-like ratio (`mean / std`) for each window, then combines two properties:

1. **Positive fraction**: What fraction of windows have positive risk-adjusted returns?
2. **Stability (1 - CV)**: How stable are the per-window Sharpe ratios (coefficient of variation)?

```
consistency = positive_frac * max(0, 1 - CV)
```

The product structure means both properties must be present: a strategy that is profitable in all windows but wildly variable still scores moderately, and a stable strategy that is consistently flat or negative also scores poorly.

**Normalization**: Already in [0, 1] from the scoring function.

#### 6. Model Attribution (10%)

Credit to L2 miners who select and combine strong L1 models. Computed externally by `ModelAttributionEngine`, this metric tracks the historical L2 performance of each L1 model and rewards miners who build strategies on models with proven deployment track records.

The attribution score for a miner is the mean of per-model quality scores across all L1 models their strategy uses:

```
per_model_score = clamp(0.5 + avg_pnl_contribution * 10, 0, 1)
miner_attribution = mean(per_model_scores)
```

- Models with no L2 history default to 0.5 (neutral).
- Models with positive average P&L contribution get scores above 0.5.
- Models with negative average P&L contribution get scores below 0.5.

This metric creates incentive alignment between L1 and L2 miners: L2 miners benefit from selecting quality models, and L1 miners benefit from having their models chosen by successful L2 strategies.

**Normalization**: Already in [0, 1] from the `ModelAttributionEngine`.

#### 7. Execution Quality (15%)

Evaluates the strategy's infrastructure health — how cleanly and efficiently it interacts with the exchange. A strategy with strong theoretical returns but poor execution (high latency, frequent rejects, excessive slippage) will degrade under real market conditions, so execution quality gates deployment readiness.

The metric combines three orthogonal sub-scores:

**Latency sub-score (40% of execution quality)**

Measures end-to-end order lifecycle speed: from signal decision through order submission, exchange acknowledgement, and fill. Uses `end_to_end_intent_ms` as the primary signal.

```
if e2e <= 200ms:  latency = 1.0
else:             latency = exp(-(e2e - 200) / 200)
```

Latency telemetry fields tracked:
- `ws_message_lag_ms` — WebSocket message lag
- `decision_to_submit_ms` — Time from decision to order submission
- `submit_to_ack_ms` — Time to exchange acknowledgement
- `ack_to_fill_ms` — Time from ack to fill
- `end_to_end_intent_ms` — Total intent execution time

**Reliability sub-score (30% of execution quality)**

Measures infrastructure stability via failure rates relative to total order volume.

```
failure_rate = (rejects + stuck + partials + reconnects) / total_orders
reliability = max(0, 1 - failure_rate * 5)
```

The 5x multiplier means a 20% failure rate zeroes the reliability sub-score. Reliability counters tracked:
- Order reject count (by reason)
- Cancel count
- Partial fill count
- Stuck order count (no response)
- Reconnect/resubscribe count

**Slippage sub-score (30% of execution quality)**

Measures realized execution cost in basis points. Lower slippage indicates better order routing, smarter sizing, and less market impact.

```
if slippage <= 5bps:  slip = 1.0
else:                 slip = exp(-(slippage - 5) / 5)
```

Performance metrics tracked:
- Slippage distribution (bps)
- Realized fees
- P&L (net of execution costs)
- Turnover

**Combined formula:**

```
execution_quality = 0.40 * latency + 0.30 * reliability + 0.30 * slippage
```

**Normalization**: Already in [0, 1] from the composite sub-score formula. Clamped as a safety guard.

#### 8. Annualized Volatility (5%)

Cumulative realized volatility of the strategy's daily returns, annualized. This is the most direct measure of how much a strategy's returns fluctuate. Strategies with high volatility carry more risk of catastrophic drawdowns and are less suitable for deployment with real capital.

```
ann_vol = std(daily_returns) * sqrt(365)
```

- Uses 365 trading days for crypto markets (24/7 operation).
- This is an **inverted** metric: lower volatility yields a higher normalized score.
- A strategy with 30% or lower annualized vol scores 1.0; at 150%+ vol, it scores 0.0.

**Normalization**: Linear interpolation: `score = clamp(1 - (vol - 0.3) / 1.2, 0, 1)`.

#### 9. Sharpe Ratio (5%)

The most widely used risk-adjusted performance measure in institutional finance. It measures excess return per unit of total volatility — answering "how much return does the strategy generate per unit of risk taken?"

```
sharpe = (mean(daily_excess_returns) / std(daily_returns)) * sqrt(365)
```

- Sharpe > 1.0 is good; > 2.0 is excellent; > 3.0 is exceptional.
- Penalizes strategies that achieve high P&L through high variance (i.e., luck-dependent returns).
- Unlike Omega ratio (which captures distribution shape), Sharpe directly penalizes the *level* of volatility.

**Normalization**: Sigmoid transform centered at 1.0: `score = 1 / (1 + exp(-1.0 * (sharpe - 1.0)))`.

#### 10. Sortino Ratio (5%)

A refinement of the Sharpe ratio that only penalizes **downside** volatility. Upside volatility (large gains) is not penalized — only the risk of losses matters. This is more appropriate for trading strategies where upside variance is desirable.

```
downside_returns = min(daily_excess_returns, 0)
downside_dev = sqrt(mean(downside_returns^2))
sortino = (mean(daily_excess_returns) / downside_dev) * sqrt(365)
```

- Values above the Sharpe ratio indicate favorable skew (more upside than downside vol).
- A strategy with high Sharpe but low Sortino has symmetric risk; high Sortino relative to Sharpe has positively skewed returns.
- Combined with Sharpe, this pair distinguishes strategies with "good volatility" (upside) from those with "bad volatility" (downside).

**Normalization**: Sigmoid transform centered at 1.5: `score = 1 / (1 + exp(-0.8 * (sortino - 1.5)))`.

### Composite Score Formula

The L2 composite score is a weighted sum of all ten normalized metrics:

```
composite = 0.17 * realized_pnl
          + 0.13 * omega
          + 0.13 * max_drawdown
          + 0.08 * win_rate
          + 0.13 * consistency
          + 0.08 * model_attribution
          + 0.13 * execution_quality
          + 0.05 * annualized_volatility
          + 0.05 * sharpe_ratio
          + 0.05 * sortino_ratio
```

Weights are published and configurable via `WeightConfig`. They are balanced so that no single metric dominates (max weight 17%), preventing single-dimension gaming.

### Why This Drives Good Behavior

- **Real outcomes only**: L2 scores are based on actual (paper/live) trading results, not simulations.
- **Drawdown elimination**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated, mirroring institutional prop trading standards.
- **Consistency requirements** prevent strategies that take one lucky trade and coast.
- **Omega ratio** captures tail risk that Sharpe ratio misses, preventing strategies that look good on average but carry hidden blow-up risk.
- **Annualized volatility** directly penalizes cumulative return fluctuation, closing a gap where strategies could achieve moderate P&L through extreme vol swings that happen to net out.
- **Sharpe and Sortino ratios** together provide a complete risk-adjusted view: Sharpe penalizes total volatility, Sortino penalizes only harmful (downside) volatility. A strategy with high upside variance but low downside deviation earns a Sortino premium over its Sharpe, correctly rewarding favorable skew.
- **Model attribution** creates incentive alignment between L1 and L2 miners.
- **Execution quality** ensures strategies are deployment-ready by penalizing high latency, infrastructure instability, and excessive slippage. A strategy with perfect returns but fragile execution will score poorly, incentivizing miners to invest in robust infrastructure.
- **Weight balance** ensures miners must optimize across all dimensions — high P&L with poor execution quality, excessive drawdown, or high volatility still scores poorly.

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
| **Defense** | Validators score against proprietary tick-by-tick data miners cannot access. Rolling holdout windows change each epoch. The proprietary overfitting detector penalizes in-sample/out-of-sample gaps. |
| **Why it fails** | The data asymmetry is the core moat. Models that only memorize public patterns will fail the proprietary benchmark. |

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
| **Defense** | Composite scoring across 7 L1 / 10 L2 metrics. No single metric dominates (max weight 22% L1, 17% L2). |
| **Why it fails** | High accuracy with high drawdown scores poorly. High Sharpe with overfitting scores poorly. High P&L with poor execution quality or high volatility scores poorly. |

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
| **Defense** | Variance Score explicitly measures cross-regime consistency. Validation windows deliberately cover trending, ranging, high-vol, low-vol, and crisis periods. Penalized F1 and Penalized Sharpe also apply rolling-window variance penalties. |
| **Why it fails** | Low Variance Score directly penalizes regime-specific models. The variance penalty in F1 and Sharpe provides a second layer of defense. |

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
