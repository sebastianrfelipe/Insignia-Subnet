# Insignia Subnet, Incentive Mechanism Design

> **Migration note:** The two independent Yuma cycles described below have been
> unified into a **single paired genetic mechanism**. Researcher and trader
> miners share one UID space and one weight vector; a candidate strategy is a
> `(researcher, trader)` pair that is jointly evaluated and ranked with NSGA-II,
> then credited via a variance-penalized marginal contribution. The 7 model
> metrics and 9 trading metrics, their weights, and the commit-reveal and
> consensus-integrity defenses documented here are all preserved, the sections
> below describe the **model** (researcher) and **trading** (trader) scoring
> components that the paired mechanism evaluates jointly. The standalone
> per-layer emission and the cross-layer feedback loop are legacy. See
> [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md).

## Overview

The Insignia incentive mechanism ensures that miners are rewarded proportionally to the genuine quality and deployability of their contributions, while making all known gaming strategies unprofitable.

Two miner roles share one subnet: **researcher miners** (who submit ML models) and **trader miners** (who run trading strategies on an assigned model). Each `(researcher, trader)` pair is jointly evaluated using the model scoring vector and the trading scoring vector documented below, then ranked and credited by the paired genetic mechanism (see [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md)).

---

## Model Scoring (Researcher Miners)

### Scoring Vector (7 Dimensions)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Penalized F1 | 22% | Directional prediction quality with cross-regime consistency penalty (mean − λ·std across windows) |
| Penalized Sharpe Ratio | 18% | Risk-adjusted returns with variance penalty across rolling sub-windows |
| Max Drawdown | 14% | Penalizes fragile models with large peak-to-trough losses |
| Variance Score | 16% | Cross-regime consistency, measures coefficient of variation across market regimes |
| Overfitting Penalty | 14% | Gap between in-sample and out-of-sample performance (proprietary metric) |
| Feature Efficiency | 6% | Penalizes models requiring exotic or excessive features |
| Latency Score | 10% | Inference speed, critical for short-horizon deployment |

All metrics use a **variance-penalized formulation** (`mean − λ·std`) across rolling windows, rewarding both peak performance and consistency.

### Why This Drives Good Behavior

- **Multi-dimensional scoring** prevents miners from gaming a single metric. A model with high F1 but 40% max drawdown scores poorly.
- **Variance-penalized metrics** ensure that high aggregate scores cannot be achieved through a single lucky window while being inconsistent elsewhere.
- **Overfitting detection** specifically targets the most common failure mode of GBDTs on financial data.
- **Variance Score** ensures models work across market regimes, not just the current one.
- **Feature efficiency** discourages models that depend on data sources that won't be available in production.

### How Model Scores Are Used

A researcher's model composite is one half of every `(researcher, trader)` pair it appears in. Pairs are ranked with NSGA-II and each miner's single Yuma emission weight is the variance-penalized marginal contribution of its model across the partners it was paired with. There is no separate promotion pool, pairing is chain-seeded by the genetic algorithm.

---

## Trading Scoring (Trader Miners)

### Scoring Vector (8 Headline Dimensions + Diagnostics)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Annualized Return | 21.28% | Scale-invariant profitability (return on capital, 365-day basis) |
| Omega Ratio | 13.83% | Full-distribution risk measure (captures tail behavior) |
| Max Drawdown | 14.89% | Hard ceiling, breach eliminates the strategy entirely |
| Consistency | 21.28% | Rolling 7-day sub-window analysis, penalizes spike-then-collapse |
| Execution Quality | 10.64% | Latency, reliability, and slippage, infrastructure health |
| Annualized Volatility | 5.32% | Cumulative realized volatility, lower = better score |
| Sharpe Ratio | 6.38% | Risk-adjusted return per unit of total volatility |
| Sortino Ratio | 6.38% | Risk-adjusted return per unit of downside volatility |

**Diagnostics tier** (computed and reported in every score vector, but **not** weighted in the composite):

| Diagnostic | Purpose |
|------------|---------|
| Win Rate | Signal precision; diagnoses churn vs. directional edge. Demoted from the headline suite — a high win rate alone does not imply profitability, and weighting it risks rewarding low-conviction noise trading. |

> **Removed, Model Attribution.** Earlier versions of this layer included a
> "Model Attribution" metric that credited a trader for the deployment track
> record of the model(s) it used. Under the single paired genetic mechanism
> the model is *assigned* to the trader by the chain-seeded genetic algorithm
> (see [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md)), not self-selected, so the
> miner has no control over which model it is paired with. Crediting that
> assignment would reward luck of the draw rather than skill, so the metric was
> removed and its weight redistributed across the remaining performance
> metrics. Cross-partner quality is now expressed structurally, through
> NSGA-II selection over pairs and the variance-penalized marginal-contribution
> credit, rather than as a per-miner scoring dimension.

### Metric Definitions

#### 1. Annualized Return (21.28%)

Measures the strategy's profitability as return on allocated capital, annualized to a 365-day crypto basis. This replaces the old absolute Realized P&L metric: raw P&L is denominated in quote currency, which made scores capital-dependent (a strategy running 10x the bankroll of another would dominate on size alone rather than skill). Return-on-capital is comparable across miners regardless of bankroll, and annualization makes scores comparable across epochs of different lengths.

```
ann_ret = (1 + cumulative_return) ** (365 / epoch_days) - 1
score   = clamp(ann_ret / 0.50, 0, 1)
```

- Non-positive annualized returns receive a score of zero (the old metric's "at or below baseline = 0" behavior).
- The linear map reaches a perfect score at 50% annualized return, exceptional for a sustained trading strategy.
- The 365-day basis matches annualized volatility (crypto markets trade 24/7).
- Carries the highest single trading weight (21.28%, tied with consistency) because realized returns are the ultimate objective of the subnet.

**Normalization**: Already in [0, 1] from the scoring function itself.

#### 2. Omega Ratio (13.83%)

A full-distribution risk measure that captures the complete shape of the return distribution, including skewness and fat tails. Unlike Sharpe ratio (which only considers mean and variance), Omega reflects the probability-weighted balance between gains and losses at a given threshold.

```
Omega = sum(max(r_i - threshold, 0)) / sum(max(threshold - r_i, 0))
```

- An Omega > 1 means the strategy's gain mass exceeds its loss mass at the threshold.
- Raw values are capped at 10.0 to prevent degenerate cases (e.g., a single winning trade with no losses) from dominating.
- This metric is critical for crypto markets, where returns are rarely normally distributed and tail risk is the primary destroyer of capital.
- **Why Omega is retained alongside Sortino**: the production execution stack assumes Student-t (fat-tailed) return innovations. Under fat tails, Sortino's downside-deviation-only view understates tail mass, while Omega integrates the full return CDF on both sides of the threshold — pricing exactly the tail behavior the Student-t assumption says will show up in production.

**Normalization**: Divided by 3.0, so Omega >= 3.0 maps to a perfect normalized score of 1.0. This threshold reflects that an Omega of 3+ is exceptional for crypto trading strategies.

#### 3. Max Drawdown (14.89%)

The peak-to-trough loss of the strategy's equity curve. This metric has a unique dual role:

1. **Soft score component**: Lower drawdown = higher normalized score (`1 - drawdown`).
2. **Hard elimination threshold**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated and receive a composite score of zero for the epoch, mirroring institutional prop trading standards.

```
drawdown = (peak_equity - current_equity) / peak_equity
normalized = 1.0 - drawdown
```

**Normalization**: Inverted so that lower drawdown = higher score. A 0% drawdown yields 1.0; a 100% drawdown yields 0.0.

#### 4. Consistency (21.28%)

Rolling sub-window analysis that penalizes "spike-then-collapse" strategies. This is the strongest predictor of a strategy's viability in live deployment.

The metric divides the return history into non-overlapping 7-day windows and computes a Sharpe-like ratio (`mean / std`) for each window, then combines two properties:

1. **Positive fraction**: What fraction of windows have positive risk-adjusted returns?
2. **Stability (1 - CV)**: How stable are the per-window Sharpe ratios (coefficient of variation)?

```
consistency = positive_frac * max(0, 1 - CV)
```

The product structure means both properties must be present: a strategy that is profitable in all windows but wildly variable still scores moderately, and a stable strategy that is consistently flat or negative also scores poorly.

**Normalization**: Already in [0, 1] from the scoring function.

#### 5. Execution Quality (10.64%)

Evaluates the strategy's infrastructure health, how cleanly and efficiently it interacts with the exchange. A strategy with strong theoretical returns but poor execution (high latency, frequent rejects, excessive slippage) will degrade under real market conditions, so execution quality gates deployment readiness.

The metric combines three orthogonal sub-scores:

**Latency sub-score (40% of execution quality)**

Measures end-to-end order lifecycle speed: from signal decision through order submission, exchange acknowledgement, and fill. Uses `end_to_end_intent_ms` as the primary signal.

```
if e2e <= 200ms:  latency = 1.0
else:             latency = exp(-(e2e - 200) / 200)
```

Latency telemetry fields tracked:
- `ws_message_lag_ms`, WebSocket message lag
- `decision_to_submit_ms`, Time from decision to order submission
- `submit_to_ack_ms`, Time to exchange acknowledgement
- `ack_to_fill_ms`, Time from ack to fill
- `end_to_end_intent_ms`, Total intent execution time

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

#### 6. Annualized Volatility (5.32%)

Cumulative realized volatility of the strategy's daily returns, annualized. This is the most direct measure of how much a strategy's returns fluctuate. Strategies with high volatility carry more risk of catastrophic drawdowns and are less suitable for deployment with real capital.

```
ann_vol = std(daily_returns) * sqrt(365)
```

- Uses 365 trading days for crypto markets (24/7 operation).
- This is an **inverted** metric: lower volatility yields a higher normalized score.
- A strategy with 30% or lower annualized vol scores 1.0; at 150%+ vol, it scores 0.0.

**Normalization**: Linear interpolation: `score = clamp(1 - (vol - 0.3) / 1.2, 0, 1)`.

#### 7. Sharpe Ratio (6.38%)

The most widely used risk-adjusted performance measure in institutional finance. It measures excess return per unit of total volatility, answering "how much return does the strategy generate per unit of risk taken?"

```
sharpe = (mean(daily_excess_returns) / std(daily_returns)) * sqrt(365)
```

- Sharpe > 1.0 is good; > 2.0 is excellent; > 3.0 is exceptional.
- Penalizes strategies that achieve high P&L through high variance (i.e., luck-dependent returns).
- Unlike Omega ratio (which captures distribution shape), Sharpe directly penalizes the *level* of volatility.

**Normalization**: Sigmoid transform centered at 1.0: `score = 1 / (1 + exp(-1.0 * (sharpe - 1.0)))`.

#### 8. Sortino Ratio (6.38%)

A refinement of the Sharpe ratio that only penalizes **downside** volatility. Upside volatility (large gains) is not penalized, only the risk of losses matters. This is more appropriate for trading strategies where upside variance is desirable.

```
downside_returns = min(daily_excess_returns, 0)
downside_dev = sqrt(mean(downside_returns^2))
sortino = (mean(daily_excess_returns) / downside_dev) * sqrt(365)
```

- Values above the Sharpe ratio indicate favorable skew (more upside than downside vol).
- A strategy with high Sharpe but low Sortino has symmetric risk; high Sortino relative to Sharpe has positively skewed returns.
- Combined with Sharpe, this pair distinguishes strategies with "good volatility" (upside) from those with "bad volatility" (downside).

**Normalization**: Sigmoid transform centered at 1.5: `score = 1 / (1 + exp(-0.8 * (sortino - 1.5)))`.

### Diagnostics (Reported, Unweighted)

These metrics are computed for every epoch and surfaced in the score vector's raw/normalized breakdowns, but carry **no weight** in the composite.

#### Win Rate

The fraction of trades that were profitable. A straightforward measure of signal precision.

```
win_rate = count(trade_pnl > 0) / total_trades
```

- Demoted from the headline suite: profitable strategies can legitimately have moderate win rates (e.g., trend-following with ~40% wins but large risk/reward ratios), and weighting it risks rewarding low-conviction noise trading.
- Its diagnostic role is filtering and forensics: strategies that generate excessive churn without directional edge are visible in the breakdown even though the composite ignores the metric.

### Composite Score Formula

The trading composite score is a weighted sum of the eight normalized headline metrics:

```
composite = 0.2128 * annualized_return
          + 0.1383 * omega
          + 0.1489 * max_drawdown
          + 0.2128 * consistency
          + 0.1064 * execution_quality
          + 0.0532 * annualized_volatility
          + 0.0638 * sharpe_ratio
          + 0.0638 * sortino_ratio
```

Weights are published and configurable via `WeightConfig`. They are balanced so that no single metric dominates (max weight 21.28%), preventing single-dimension gaming. The 2026-08-03 renormalization scaled the eight surviving weights pro-rata (x 1/0.94) after win rate moved to the diagnostics tier, preserving their relative proportions.

### Why This Drives Good Behavior

- **Real outcomes only**: trading scores are based on actual (paper/live) trading results, not simulations.
- **Scale-invariant profitability**: annualized return rewards skill per unit of capital, not bankroll size, and stays comparable across epoch lengths.
- **Drawdown elimination**: Strategies that breach the drawdown limit (default 20%) are immediately eliminated, mirroring institutional prop trading standards.
- **Consistency requirements** prevent strategies that take one lucky trade and coast.
- **Omega ratio** captures tail risk that Sharpe ratio misses, preventing strategies that look good on average but carry hidden blow-up risk. It is retained alongside Sortino specifically because the production stack assumes Student-t tails: Omega prices the full return distribution, Sortino only its downside deviation.
- **Annualized volatility** directly penalizes cumulative return fluctuation, closing a gap where strategies could achieve moderate P&L through extreme vol swings that happen to net out.
- **Sharpe and Sortino ratios** together provide a complete risk-adjusted view: Sharpe penalizes total volatility, Sortino penalizes only harmful (downside) volatility. A strategy with high upside variance but low downside deviation earns a Sortino premium over its Sharpe, correctly rewarding favorable skew.
- **No reward for the assigned partner.** A trader is not credited or penalized for the deployment track record of the model it was paired with, because that pairing is assigned by the chain-seeded genetic algorithm and is outside the miner's control. Cross-partner model quality surfaces structurally through NSGA-II pair selection and the variance-penalized marginal-contribution credit, not as a per-miner scoring dimension.
- **Execution quality** ensures strategies are deployment-ready by penalizing high latency, infrastructure instability, and excessive slippage. A strategy with perfect returns but fragile execution will score poorly, incentivizing miners to invest in robust infrastructure.
- **Weight balance** ensures miners must optimize across all dimensions, high return with poor execution quality, excessive drawdown, or high volatility still scores poorly.

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

### 3. Model Plagiarism (Researcher)

| | |
|---|---|
| **Attack** | Miner copies another miner's model artifact or reverse-engineers their approach. |
| **Defense** | SHA-256 fingerprinting detects exact duplicates. Prediction correlation analysis detects behavioral clones. Correlated models share rewards. |
| **Why it fails** | No incentive to copy, you only get a fraction of the reward. Original work pays more. |

### 4. Copy-Trading (Trader)

| | |
|---|---|
| **Attack** | Trader miner mirrors another miner's positions instead of building their own strategy. |
| **Defense** | Position correlation analysis with time/size tolerance. Correlated strategies share rewards. |
| **Why it fails** | Same as model plagiarism, copying dilutes your reward. |

### 5. Single-Metric Gaming

| | |
|---|---|
| **Attack** | Miner optimizes for one dominant metric while ignoring others. |
| **Defense** | Composite scoring across 7 model / 9 trading metrics. No single metric dominates (max weight 22% model, 20% trading). |
| **Why it fails** | High accuracy with high drawdown scores poorly. High Sharpe with overfitting scores poorly. High P&L with poor execution quality or high volatility scores poorly. |

### 6. Validator Data Leakage

| | |
|---|---|
| **Attack** | Miner reverse-engineers the proprietary validation dataset from score feedback. |
| **Defense** | Only aggregate scores returned, never raw predictions or data. Rolling windows change each epoch. Historical windows released 30 days after evaluation. |
| **Why it fails** | Aggregate scores reveal almost no information about the underlying data distribution. |

### 7. Trader Paper Trading Manipulation

| | |
|---|---|
| **Attack** | Trader miner fabricates paper trading results or cherry-picks favorable reporting windows. |
| **Defense** | Validators track positions via continuous streaming. All positions are timestamped. Reporting gaps are penalized. |
| **Why it fails** | Validators independently verify position state, fabricated results are immediately detected. |

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

### 13. Researcher/Trader Weight Skew Exploitation (legacy)

| | |
|---|---|
| **Attack** | (Legacy two-layer attack.) Adversarial miners exploit the emission split between the model and trading layers to capture disproportionate rewards by concentrating effort in the more rewarding layer. The single paired mechanism has no emission split, both roles share one weight vector, so this surface no longer exists. |
| **Defense** | `cross_layer_penalty_strength` penalizes deviations from the configured `l1_l2_emission_split`. Cross-layer feedback ensures both layers must perform well. |
| **Why it fails** | The penalty is proportional to the deviation, making exploitation unprofitable. |

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

This creates a direct link between the subnet's economic output and miner token value, an alignment mechanism that existing trading subnets lack.

---

## Deployment Collateral & Loss-Linked Slashing (End State)

**Design commitment: burn, don't just withhold.** Every defense above operates by *withholding* — a gamed submission scores zero and forfeits emissions it never had. The signal-driven adversary penalties (this branch) sharpen that, but penalty scoring alone leaves an asymmetry: a deployed pair that loses the desk real money has, at worst, lost future upside. The end state closes that asymmetry with real economic downside: **deployed pairs post staked alpha as collateral against their live P&L, and realized losses slash the collateral.** Scoring penalties remain the interim and screening layer; slashing is the explicit end state for the deployment tier.

### Mechanism

1. **Staked-to-participate.** Acceptance into the deployment pipeline (the top-pair tier the desk actually trades) requires the pair's miners to post an alpha bond, escrowed via `transfer_stake` to a fund-controlled collateral coldkey. Bond size scales with allocated deployment capital. Undeployed pairs are unaffected — this gates the tier where miner output touches real money.
2. **Loss-linked slashing.** Realized losses attributable to a deployed pair (net, over the settlement window) slash the bond up to a per-window cap. Realized *gains* accrue the pair's standard deployment rewards; the bond is returned (plus accrued staking emissions) on clean undeployment. **How the slash is split between the two miners is a first-order design question — see §Splitting a slash below; it is not pro-rata by bond size.**
3. **Slashed alpha is burned, not redistributed.** Redistribution creates a bounty for inducing other pairs' losses and is recyclable by sybil clusters; a burn is the only sink no adversary can route back to themselves. Burns are what separate signal from noise: a miner who won't stake against their own live P&L is telling you their signal isn't one.

### Splitting a slash between two unaffiliated miners

A `(researcher, trader)` pair is two **separate** miners, and §2.3 of [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md) assigns them to each other deterministically from chain block hash, hiding partner identity until evaluation. Neither miner can screen or monitor the other. A slash split by bond size alone would therefore punish a miner for a partner's error with no channel through which they could have prevented it — and would reintroduce precisely the partner noise the emission side is engineered to remove (the K-partner floor plus the variance-penalized `mean − λ·std` credit formula in `pairing.py::MarginalContributionCredit`).

The evolutionary mechanism does **not** launder this away, for three reasons: the K-partner averaging that de-noises credit does not exist at the deployment tier (a pair is deployed as one specific pair, over a settlement window of weeks, a handful of times); emissions are a recoverable flow while the bond is a depleting stock with an absorbing floor, so variance can eliminate a miner before the mean arrives; and joint liability earns its keep through peer screening and monitoring, both of which assigned pairing forecloses.

Slashes are therefore split by **attribution** (`treasury/collateral.py::blame_split`):

| Portion of the loss | Determined by | Who pays |
|---|---|---|
| **Explained** — `min(1, d_researcher + d_trader)` | Per-role diagnostic degradation from validation to live: `overfitting_penalty` / `penalized_f1` / `variance` for the researcher, `execution_quality` / `consistency` / `penalized_sharpe` for the trader | The degraded role, in proportion to its own degradation |
| **Unexplained** — the residual | Nothing in either role's diagnostics accounts for it | Shared pro-rata by bond, but only `ambiguous_exposure` (default 50%) of it is slashed at all |

Two properties worth stating explicitly. First, joint liability is reduced, **not eliminated**: a sound model and a sound strategy can still be a bad pairing (a high-turnover strategy on a slow-decaying signal), and that joint-mismatch cost is real — driving attribution to 100% would remove any incentive to be robust across partners, which is exactly what the variance penalty in the credit formula rewards. Second, punishment scales with the strength of the justification, and the unslashed remainder is simply **forgiven**: the bond is an incentive device, not a loss-recovery claim against miners, so a loss nobody can explain is a weak basis for taking someone's stake. Missing diagnostics therefore reduce the slash rather than defaulting to blame.

### Settlement pipeline (chain mechanics)

The chain has no native "slash a miner" primitive — the slash is enforced at the escrow layer (the bond sits on a fund coldkey under the deployment agreement), and the *burn* leg uses the subnet-owner burn extrinsic:

- **Slash leg:** unstake the slashed alpha from escrow (sells into the pool → TAO proceeds).
- **Burn leg:** the subnet owner calls `add_stake_burn` with those TAO proceeds. TAO is withdrawn from the owner coldkey into the pool's TAO reserve; the AMM prices the equivalent alpha, which is removed from the alpha reserve and burned in the same transaction. Net effect of both legs: circulating alpha falls by ≈ the slashed amount (less fees/slippage), pool TAO roughly round-trips, and alpha price adjusts upward from the supply reduction.

Operational constraints (verify on-chain before implementing, per SPEC §0 discipline):

- **Rate-limited: one `add_stake_burn` per tempo per subnet** (`AddStakeBurnRateLimitExceeded` on violation; default tempo 360 blocks ≈ 72 min at 12 s blocks). Slash settlement is therefore **batched per tempo** — which also matches the epoch cadence of scoring and keeps burns predictable and auditable.
- **Slippage applies — but to the legs, not the round trip.** Because the burn leg re-buys on the pool the slash leg just displaced, the two legs' slippage cancels: net supply cost is ≈ 2× the pool fee at any batch size (verified in `tests/test_burn_settlement.py`). What does scale with size is the *transient* price displacement between the legs, which is front-runnable — so set an explicit limit price on both legs and split batches to bound the slash leg's price impact (implemented in `treasury/execution/burn.py::plan_settlement`).
- The owner may equivalently fund the burn leg from treasury TAO and retain the slashed alpha in inventory — identical supply effect, different treasury composition; treat as a routing-policy choice (SPEC §5).

### Why this is also a retention lever

Bonded alpha is alpha that cannot be sold while the pair is deployed — the deployment tier, which earns the most, is exactly the cohort whose sell-through matters most. This directly attacks the miner sell-through σ in RISK_REGISTER R11 and compounds the existing retention levers (deployment pipeline access, token-gated API). Report bonded collateral and cumulative burned alpha in the monthly retention metrics (SPEC §8).

**Legal note:** collateral posting and slashing terms for miners are contractual; route the deployment-agreement terms through Phase-0 counsel alongside the LP documents (they are miner-facing, not investor-facing, so they do not sit behind the `LEGAL_SIGNOFF` gate — but they are enforceable agreements and need the same review).

---

## Economic Sustainability

| Revenue Stream | Description |
|---|---|
| **Internal Deployment** (Primary) | Subnet owner's firm deploys winning pairs in live prop trading. Direct P&L justifies infrastructure costs. |
| **Emissions Bootstrap** | Bittensor network emissions fund miner participation before external revenue scales. |
| **Buyback Loop** | Deployment profits → token buybacks → higher miner incentives → better models. |
| **External Signal API** (Future) | Package trading strategy outputs as subscription service for external quant funds. |
| **Model Marketplace** (Future) | License winning model architectures to external ML teams. |

The subnet owner being the primary consumer of trader output creates a demand floor that does not depend on external market adoption.
