# Insignia Subnet — Incentive Mechanism Design

> **Migration note:** The two independent Yuma cycles described below are being
> unified into a **single paired genetic mechanism**. Researcher and trader
> miners share one UID space and one weight vector; a candidate strategy is a
> `(researcher, trader)` pair that is jointly evaluated and ranked with NSGA-II,
> then credited via a variance-penalized marginal contribution. The 7 model
> metrics and 10 trading metrics, their weights, and the commit-reveal and
> consensus-integrity defenses documented here are all preserved. See
> [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md).

## Overview

The Insignia incentive mechanism ensures that miners are rewarded proportionally to the genuine quality and deployability of their contributions, while making all known gaming strategies unprofitable.

The design operates as a single mechanism: researcher and trader miners share one Yuma consensus cycle, matched into `(model, strategy)` pairs that are jointly evaluated and selected with a genetic algorithm.

---

## Pairing & Joint Fitness

A candidate strategy is a `(researcher, trader)` **pair**. The validator forms
chain-seeded pairs (each miner placed in `K = partners_per_miner` pairs), then
scores each pair on the same metrics documented below:

- the **researcher half** is the 7-metric model score (`score_l1`),
- the **trader half** is the 10-metric trading score (`score_l2`) from running the
  trader's strategy on the paired model,
- combined as `pair_composite = pair_blend_alpha * model + (1 - pair_blend_alpha) * trading`
  plus a 4-objective fitness vector `[-model, -trading, drawdown, -consistency]`.

The metric definitions and weights below are unchanged from the two-layer design;
only their *combination and selection* changed. See
[PAIRING_MECHANISM.md](PAIRING_MECHANISM.md).

## Researcher Half: Model Quality Incentives

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

Researchers and traders share a single Yuma weight vector. A miner's weight is its
variance-penalized marginal contribution across the pairs it participated in (see
"Pairing & Genetic Selection" below) — there is no separate L1 emission pool and
no `l1_l2_emission_split`.

---

## Trader Half: Live Trading Incentives

### Scoring Vector (10 Dimensions)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Realized P&L | 21% | Absolute returns from actual trading outcomes |
| Omega Ratio | 15% | Full-distribution risk measure (captures tail behavior) |
| Max Drawdown | 12% | Hard ceiling — breach eliminates the strategy entirely |
| Win Rate | 7% | Signal precision — penalizes low-conviction noise trading |
| Consistency | 17% | Rolling 7-day sub-window analysis — penalizes spike-then-collapse |
| Model Attribution | 11% | Credit for the assigned model's trading track record |
| Execution Quality | 5% | Latency, reliability, and slippage — infrastructure health |
| Annualized Volatility | 5% | Cumulative realized volatility — lower = better score |
| Sharpe Ratio | 5% | Risk-adjusted return per unit of total volatility |
| Sortino Ratio | 5% | Risk-adjusted return per unit of downside volatility |

### Metric Definitions

#### 1. Realized P&L (21%)

Measures the strategy's raw profitability relative to a baseline (typically buy-and-hold or zero). This is the most direct measure of whether a strategy generates economic value.

```
score = clamp((pnl - baseline) / max(|baseline|, 1.0), 0, 1)
```

- Strategies at or below the baseline receive a score of zero.
- The denominator scales by the absolute baseline value so that the metric is meaningful across different capital levels and market conditions.
- Carries one of the highest single trading weights because realized returns are the ultimate objective of the subnet.

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

#### 3. Max Drawdown (12%)

The peak-to-trough loss of the strategy's equity curve. This metric has a unique dual role:

1. **Soft score component**: Lower drawdown = higher normalized score (`1 - drawdown`).
2. **Hard elimination threshold**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated and receive a composite score of zero for the epoch, mirroring institutional prop trading standards.

```
drawdown = (peak_equity - current_equity) / peak_equity
normalized = 1.0 - drawdown
```

**Normalization**: Inverted so that lower drawdown = higher score. A 0% drawdown yields 1.0; a 100% drawdown yields 0.0.

#### 4. Win Rate (7%)

The fraction of trades that were profitable. A straightforward measure of signal precision.

```
win_rate = count(trade_pnl > 0) / total_trades
```

- Carries the lowest weight (10%) by design, because profitable strategies can legitimately have moderate win rates (e.g., trend-following with ~40% wins but large risk/reward ratios).
- Its primary role is to filter out noise trading: strategies that generate excessive churn without directional edge.

**Normalization**: Already in [0, 1] from the scoring function.

#### 5. Consistency (17%)

Rolling sub-window analysis that penalizes "spike-then-collapse" strategies. This is the strongest predictor of a strategy's viability in live deployment.

The metric divides the return history into non-overlapping 7-day windows and computes a Sharpe-like ratio (`mean / std`) for each window, then combines two properties:

1. **Positive fraction**: What fraction of windows have positive risk-adjusted returns?
2. **Stability (1 - CV)**: How stable are the per-window Sharpe ratios (coefficient of variation)?

```
consistency = positive_frac * max(0, 1 - CV)
```

The product structure means both properties must be present: a strategy that is profitable in all windows but wildly variable still scores moderately, and a stable strategy that is consistently flat or negative also scores poorly.

**Normalization**: Already in [0, 1] from the scoring function.

#### 6. Model Attribution (11%)

Credit for the quality of the model the trader is paired with. Under the single
paired mechanism the trader does not self-select models — the validator assigns a
researcher partner — so this metric tracks the historical trading performance of
the assigned model and rewards pairs built on models with proven track records.

The attribution score is the mean of per-model quality scores across the assigned model(s):

```
per_model_score = clamp(0.5 + avg_pnl_contribution * 10, 0, 1)
miner_attribution = mean(per_model_scores)
```

- Models with no trading history default to 0.5 (neutral).
- Models with positive average P&L contribution get scores above 0.5.
- Models with negative average P&L contribution get scores below 0.5.

This complements the pairing engine: pairing exposes each model to many traders,
and attribution rewards the pairs that turn good models into real PnL.

**Normalization**: Already in [0, 1] from the `ModelAttributionEngine`.

#### 7. Execution Quality (5%)

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

The trading composite score is a weighted sum of all ten normalized metrics:

```
composite = 0.21 * realized_pnl
          + 0.15 * omega
          + 0.12 * max_drawdown
          + 0.07 * win_rate
          + 0.17 * consistency
          + 0.08 * model_attribution
          + 0.05 * execution_quality
          + 0.05 * annualized_volatility
          + 0.05 * sharpe_ratio
          + 0.05 * sortino_ratio
```

Weights are published and configurable via `WeightConfig`. They are balanced so that no single metric dominates (max weight 17%), preventing single-dimension gaming.

### Why This Drives Good Behavior

- **Real outcomes only**: trading scores are based on actual (paper/live) trading results, not simulations.
- **Drawdown elimination**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated, mirroring institutional prop trading standards.
- **Consistency requirements** prevent strategies that take one lucky trade and coast.
- **Omega ratio** captures tail risk that Sharpe ratio misses, preventing strategies that look good on average but carry hidden blow-up risk.
- **Annualized volatility** directly penalizes cumulative return fluctuation, closing a gap where strategies could achieve moderate P&L through extreme vol swings that happen to net out.
- **Sharpe and Sortino ratios** together provide a complete risk-adjusted view: Sharpe penalizes total volatility, Sortino penalizes only harmful (downside) volatility. A strategy with high upside variance but low downside deviation earns a Sortino premium over its Sharpe, correctly rewarding favorable skew.
- **Model attribution** rewards pairs built on models with proven trading track records.
- **Execution quality** ensures strategies are deployment-ready by penalizing high latency, infrastructure instability, and excessive slippage. A strategy with perfect returns but fragile execution will score poorly, incentivizing miners to invest in robust infrastructure.
- **Weight balance** ensures miners must optimize across all dimensions — high P&L with poor execution quality, excessive drawdown, or high volatility still scores poorly.

---

## Pairing & Genetic Selection

Joint evaluation replaces promotion and retroactive feedback. Each generation:

```
Researchers + Traders ──> chain-seeded pairing ──> joint pair evaluation
        ▲                                                   │
        │                                                   ▼
   crossover + mutation ◄── NSGA-II rank + marginal credit ◄┘
```

1. **Chain-seeded pairing** (`ChainSeededPairing`): pairs are derived
   deterministically from chain block state; neither miners nor validators choose
   partners, and partner identity is revealed only at evaluation time. Every miner
   appears in at least `partners_per_miner` (K) pairs.
2. **NSGA-II selection** (`NSGA2Matchmaker`): pairs are ranked by non-dominated
   sorting + crowding over the 4-objective fitness, blended with the scalar
   `pair_composite` so quality still discriminates on crowded Pareto fronts.
3. **Collusion screen** (`CollusionGraphDetector`): a pair whose composite far
   exceeds *each* partner's best alternative pairing (non-transferable lift) is
   flagged and its contribution discounted.
4. **Marginal-contribution credit** (`MarginalContributionCredit`): a miner's
   weight is `mean − marginal_contribution_weight · std` of its pair selection
   scores across partners, normalized over all miners to one Yuma vector.
   Transferable skill (good across many partners) is rewarded; one-hit / collusive
   pairings are demoted.
5. **Reproduction**: rank-0 elites survive; elite researchers and traders are
   recombined (crossover) and randomly re-paired (mutation) for the next
   generation.

This closes the simulation-to-reality gap *jointly*: a model is only valuable if
some trader can turn it into real PnL, and a trader is only valuable if it can do
so across the models it is assigned.

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

### 3. Model Plagiarism (researcher)

| | |
|---|---|
| **Attack** | Miner copies another miner's model artifact or reverse-engineers their approach. |
| **Defense** | SHA-256 fingerprinting detects exact duplicates. Prediction correlation analysis detects behavioral clones. Correlated models share rewards. |
| **Why it fails** | No incentive to copy — you only get a fraction of the reward. Original work pays more. |

### 4. Copy-Trading (trader)

| | |
|---|---|
| **Attack** | A trader miner mirrors another miner's positions instead of building their own strategy. |
| **Defense** | Position correlation analysis with time/size tolerance. Correlated strategies share rewards. |
| **Why it fails** | Same as model plagiarism — copying dilutes your reward. |

### 5. Single-Metric Gaming

| | |
|---|---|
| **Attack** | Miner optimizes for one dominant metric while ignoring others. |
| **Defense** | Composite scoring across 7 model / 10 trading metrics. No single metric dominates (max weight 22% model, 18% trading). |
| **Why it fails** | High accuracy with high drawdown scores poorly. High Sharpe with overfitting scores poorly. High P&L with poor execution quality or high volatility scores poorly. |

### 6. Validator Data Leakage

| | |
|---|---|
| **Attack** | Miner reverse-engineers the proprietary validation dataset from score feedback. |
| **Defense** | Only aggregate scores returned, never raw predictions or data. Rolling windows change each epoch. Historical windows released 30 days after evaluation. |
| **Why it fails** | Aggregate scores reveal almost no information about the underlying data distribution. |

### 7. Paper Trading Manipulation (trader)

| | |
|---|---|
| **Attack** | A trader miner fabricates paper trading results or cherry-picks favorable reporting windows. |
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

### 10. Post-Hoc Prediction Manipulation (Validator Latency Exploitation)

| | |
|---|---|
| **Attack** | Miner submits trades using market data that has already materialized but not yet been validated by a slow validator, achieving artificially high accuracy during high-latency windows. |
| **Defense** | Commit-reveal scheme requires miners to commit trade hashes before market data is available. `min_prediction_lead_time` rejects trades where submission is too close to data publication. `validator_latency_penalty_weight` discounts scores from high-latency validators. Three detection methods: per-validator latency correlation, submission vs market timestamp comparison, quartile-segmented accuracy analysis. |
| **Why it fails** | Commit-reveal eliminates the information asymmetry: predictions are cryptographically bound before market data exists. Sentinel validation confirms projected severity drops from 0.09 to 0.047 with commit-reveal (below 0.05 target). |

### 11. Prediction Timing Manipulation

| | |
|---|---|
| **Attack** | Miner exploits timing gaps between prediction submission and validation to incorporate information that should not have been available at prediction time. |
| **Defense** | Commit-reveal binds predictions to a specific point in time (commit window closes at T-5s before market data). Temporal correlation monitoring detects miners whose submission patterns change in lockstep with market movements. |
| **Why it fails** | With commit-reveal, the prediction is locked before market data is published. Severity drops from 0.06 to projected 0.025. |

### 12. Miner-Validator Collusion

| | |
|---|---|
| **Attack** | A colluding validator inflates scores for cooperating miners via weight-setting manipulation, score inflation, or information leakage. |
| **Defense** | Weight entropy minimum rejects concentrated weight distributions. Cross-validator score variance caps flag miners with inconsistent scores across validators. Validator rotation limits prevent repeated scoring of the same miner. Five detection methods: weight entropy analysis, cross-validator score comparison, weight-non-performance correlation, temporal coordination, network graph cluster analysis. |
| **Why it fails** | Multi-validator consensus means a single colluding validator cannot unilaterally inflate scores. The 5-method detection approach catches different collusion strategies. |

### 13. Pair Collusion (researcher ↔ trader)

| | |
|---|---|
| **Attack** | A researcher and trader privately cooperate so their `(model, strategy)` pair scores well only when matched together (non-transferable lift), capturing emissions without genuine partner-independent quality. |
| **Defense** | Chain-seeded pairing prevents self-selecting an accomplice; the K-partner floor forces evaluation against partners the miner did not choose; variance-penalized marginal credit erodes one-hit pairings; `CollusionGraphDetector` flags pairs whose composite far exceeds each partner's best alternative pairing. |
| **Why it fails** | Both ring members underperform with everyone else, so their cross-partner mean is mediocre and their variance is high — the credit `mean − λ·std` collapses, and the flagged pair's contribution is discounted. |

### 14. Partner-Selection Gaming

| | |
|---|---|
| **Attack** | A miner tries to steer which partner it is matched with (timing registration/submissions) to secure a favorable counterpart. |
| **Defense** | Pairing is deterministic from chain block state and partner identity is revealed only at evaluation time, so there is no controllable selection surface; reproduction re-matches elites across generations. |
| **Why it fails** | The assignment depends on the chain block hash, which the miner cannot predict or influence. |

### 15. Latency Arbitrage in Pairing

| | |
|---|---|
| **Attack** | A miner exploits validator latency or partner foreknowledge to submit model/trade decisions after benchmark/market data has materialized. |
| **Defense** | Commit-reveal binds both the model and the trade commitments before the window; partner identity is unknown until evaluation; `min_prediction_lead_time` and `validator_latency_penalty_weight` discount late or high-latency submissions. |
| **Why it fails** | Commitments are cryptographically fixed before data exists and there is no known partner/validator to time against. |

---

## Commit-Reveal Mechanism

### Overview

The commit-reveal scheme prevents post-hoc prediction manipulation (Vector 8) and prediction timing manipulation (Vector 11) by requiring miners to cryptographically commit to their trade decisions before market data is available, then reveal after the validation window closes.

**Implementation:** Approach B (off-chain with validator attestation), implemented in `CommitRevealManager` in `insignia/incentive.py`.

### Protocol Flow

```
T-35s ──── Commit Window Opens ────────── T-5s
                                            │
  Miner: hash = SHA-256(trade_data ∥ nonce) │
  Miner: submit commit_hash to validators   │
  Validators: attest to receiving commit     │
                                            │
T-5s ────── Commit Window Closes ──────── T+0s
                                            │
  Market data published (T+0s)               │
                                            │
T+5s ────── Reveal Window Opens ──────── T+20s
                                            │
  Miner: reveal trade_data + nonce           │
  Validator: recompute hash, verify match    │
  Validator: score trade (or zero if invalid)│
                                            │
T+20s ───── Reveal Window Closes ─────────
```

### Technical Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Hash algorithm | SHA-256 | Commitment binding |
| Nonce size | 128-bit (16 bytes) | Prevent rainbow table attacks |
| Commit window | 30s (T-35s to T-5s) | Time for miners to commit |
| Reveal window | 15s (T+5s to T+20s) | Time for miners to reveal |
| Grace period | 2s | Clock skew tolerance |
| Late reveal penalty | 1.0 (full score zeroed) | Enforce reveal discipline |

### Sentinel Validation Results (2026-04-12)

The sentinel agent validated that commit-reveal reduces Vector 8 severity below the 0.05 target:

```
projected_severity = base_severity × (1 - effectiveness) + residual
                   = 0.09 × (1 - 0.70) + 0.02
                   = 0.047  ✓  (< 0.05 target)
```

| Metric | Value |
|--------|-------|
| Vector 8 current severity | 0.09 |
| Projected severity with commit-reveal | 0.047 |
| Target | 0.05 |
| Safety margin | 0.003 (6%) |
| Attack surface eliminated | 0.063 |
| Residual attack surface | 0.02 |

**Sensitivity:** Commit-reveal effectiveness must exceed 0.667 to meet the target. The third orchestration run strengthened the live operating margin to `0.76`, while the simulator's pre/post validation study measured `0.801` effectiveness across 25 pre-CR and 25 post-CR epochs. The earlier `0.700` / 6-validation sentinel posture and the harsher `0.723` benchmark remain useful historical gates, but the current state is materially stronger and should now be treated as a validation baseline before production deployment.

**Bonus:** Vector 11 (Prediction Timing Manipulation) drops from 0.06 to projected 0.025.

### Deployment Strategy

The commit-reveal mechanism uses a hybrid deployment:

1. **Phase 1 (Months 1-3):** Optional for miners. Miners who use commit-reveal receive a small scoring bonus. Non-committing miners are still scored normally.
2. **Phase 2 (Month 3+):** Mandatory for all miners. Submissions without valid commit-reveal are scored zero.
3. **Future (Approach C):** Migrate to hybrid on-chain reveal where reveal hashes are anchored to chain state, providing cryptographic guarantees against selective revelation.

### Residual Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Strategic commitment avoidance | 0.008 | `commitment_violation_score` in Vector 8 detection |
| Pre-commit data snooping | 0.005 | Commit window closes at T-5s, before market data |
| Selective revelation | 0.004 | No-reveal slashing: 3-consecutive penalty zeroes score |
| Validator collusion on commits | 0.003 | Multi-validator attestation + hash binding |

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
| **External Signal API** (Future) | Package winning pair strategy outputs as a subscription service for external quant funds. |
| **Model Marketplace** (Future) | License winning model architectures to external ML teams. |

The subnet owner being the primary consumer of the winning pairs' trading output creates a demand floor that does not depend on external market adoption.
