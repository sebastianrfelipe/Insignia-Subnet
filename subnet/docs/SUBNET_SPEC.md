# Insignia Subnet Specification

**Version**: 0.1.0
**Category**: Quantitative Finance / DeFi
**Track**: Bittensor Sovereign Infrastructure Hackathon

> **Architecture:** Insignia uses a **single paired genetic incentive mechanism**
> — researcher and trader miners matched into `(model, strategy)` pairs, jointly
> evaluated, and selected with NSGA-II. The 7 model + 10 trading metrics and their
> weights are preserved from the earlier two-layer design; the L1→promotion→L2
> flow, the cross-layer feedback loop, and `l1_l2_emission_split` are gone,
> replaced by chain-seeded pairing, joint pair evaluation, and
> marginal-contribution credit. See [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md)
> for the authoritative design.

---

## 1. Subnet Identity

| Field | Value |
|-------|-------|
| **Name** | Insignia |
| **Tagline** | Decentralized Predictive Modeling for On-Chain Markets |
| **Architecture** | Single paired genetic mechanism (researcher + trader miners matched into pairs) |
| **Primary Output** | Battle-tested ML model + trading strategy pairs |
| **Target Instruments** | BTC, ETH, SOL, AVAX, ADA perpetual futures |
| **Time Horizon** | Short-horizon (minutes to hours) |

---

## 2. Build Requirements Coverage

### Benchmark
**Target output quality metric miners are scored on.**

A candidate strategy is a `(researcher, trader)` **pair**. Each pair is scored on
the same metrics as before, now combined into one joint fitness:

- **Model half (researcher)**: 7 dimensions (Penalized F1, Penalized Sharpe, Max Drawdown, Variance Score, Overfitting Penalty, Feature Efficiency, Latency) scored against the proprietary tick-by-tick benchmark dataset
- **Trading half (trader)**: 10 dimensions (Realized P&L, Omega Ratio, Max Drawdown, Win Rate, Consistency, Model Attribution, Execution Quality, Annualized Volatility, Sharpe Ratio, Sortino Ratio) measured by running the trader's strategy on the paired model, with commit-reveal preventing post-hoc manipulation
- The two are combined into a `pair_composite` plus a multi-objective fitness vector (`combine_pair_scores` in `scoring.py`); published metric definitions with configurable weights are unchanged

### Evaluation Loop
**How validators score miners; cost, scale consideration.**

- Each pair: vectorized model backtest (O(n)) + a trading run on that model (O(steps)); parallelizable across pairs
- Each miner is evaluated against `K = partners_per_miner` partners (chain-seeded), so cost scales linearly with the population, not quadratically
- Validators converge deterministically (same scorer + same chain-seeded pairing + same data = same scores)
- NSGA-II non-dominated sorting + crowding ranks pairs; a variance-penalized marginal contribution turns pair ranks into one per-miner weight vector

### Miner Task
**Specific, measurable, implementable task interface.**

- **Researcher Task**: Train and submit ML model artifacts (ONNX/joblib) for directional prediction
  - Interface: `L1ModelSubmission` / `PairEvaluationRequest` with `role = researcher` (see `protocol.py`)
  - Measurable via the model score vector inside the pair
- **Trader Task**: Build and operate a paper/live trading strategy on the validator-assigned model
  - Interface: `PairEvaluationRequest` with `role = trader`; returns strategy params + position log
  - Measurable via realized P&L, Omega, drawdown, consistency, execution quality of the pair

### Incentive Design
**Why scoring rewards genuine quality; why top attack vectors fail.**

- Composite scoring across model + trading metrics prevents single-metric gaming
- Data asymmetry (miners: public data; validators: proprietary data) prevents overfitting to validation
- Joint pair evaluation rewards models that translate into real trading outcomes and traders that exploit good models — no promotion bottleneck, no `l1_l2_emission_split`
- Variance-penalized marginal-contribution credit rewards transferable skill (works across many partners) and demotes one-hit / collusive pairings
- Chain-seeded pairing removes miner self-selection and partner foreknowledge (anti-collusion, anti-latency-arbitrage)
- Commit-reveal binds both model and trade commitments before the window (SHA-256 + 128-bit nonces, commit window T-35s to T-5s, reveal window T+5s to T+20s)
- The post-commit-reveal operating model is evaluated against 28 surveillance vectors, including 3 paired-mechanism vectors (`pair_collusion`, `partner_selection_gaming`, `latency_arbitrage_pairing`)
- The latest orchestration run validated commit-reveal effectiveness at `0.76`, above the `0.667` acceptance floor
- NSGA-II v13 R2 hit the primary target with breach_rate `3.5e-6`, honest_score `0.9795`, and separation `0.953`

### Market Demand
**Who pays for output and why.**

- **Primary buyer**: Subnet owner's prop trading firm (built-in demand floor)
- **Value proposition**: Model + strategy pairs with dual-layer performance proof (simulation + live)
- **Future buyers**: External quant funds seeking validated trading strategies

### Sovereignty Test
**Subnet survives if any single cloud, company, API disappears.**

- Researcher miners run their own compute — no cloud dependency
- Validators can fall back to premium public data sources
- Trader paper trading can switch exchange price feeds trivially
- Core architecture (chain-seeded paired evaluation + genetic selection) survives any single provider outage

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INSIGNIA SUBNET                              │
│                  (single UID space, one weight vector)               │
│                                                                      │
│   ⛏ Researcher miners            📈 Trader miners                     │
│   (ML models)                    (trading operations)               │
│        │                               │                            │
│        └────────────┬──────────────────┘                            │
│                     ▼                                                │
│        Chain-seeded pairing  →  population of (researcher, trader)   │
│                     │            pairs (genomes)                     │
│                     ▼                                                │
│        Joint evaluation: model on benchmark + strategy on that model │
│                     ▼                                                │
│        NSGA-II non-dominated sort + crowding (multi-objective)       │
│                     ▼                                                │
│        Collusion screen + variance-penalized marginal credit        │
│                     ▼                                                │
│        Single Yuma set_weights  ──►  🏢 Firm deploys winning pairs   │
│                     │                       │                        │
│        Crossover + mutation (next gen)      └─► Buyback → token value │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow (one generation)

1. Researcher miners submit model artifacts; trader miners submit strategies
2. The validator forms chain-seeded `(researcher, trader)` pairs (each miner in ≥ K pairs)
3. Each pair is jointly evaluated: model on the proprietary benchmark + the trader's strategy run on that model
4. NSGA-II ranks pairs by Pareto front + crowding over the multi-objective fitness
5. A collusion-graph screen flags non-transferable lift; a variance-penalized marginal contribution converts pair ranks to one per-miner weight vector
6. Elite pairs are bred (crossover + mutation) for the next generation; the subnet owner deploys winning model+strategy pairs

---

## 4. Technical Interfaces

Miners declare a `role` (`researcher` or `trader`) at registration. The validator
announces the chain-seeded pairing via `PairAssignment` and requests artifacts via
`PairEvaluationRequest`.

### Researcher Model Submission

```python
class L1ModelSubmission(bt.Synapse):
    role: str = "researcher"
    model_artifact: bytes          # Serialized model (ONNX/joblib, max 50MB)
    model_type: str                # gbdt | neural | ensemble | other
    features_used: List[str]       # From published feature registry
    training_window_start: str     # ISO datetime
    training_window_end: str       # ISO datetime
    hyperparams: Dict[str, Any]    # Full hyperparameter set
    target_instrument: str         # e.g., "BTC-USDT-PERP"
    target_horizon_minutes: int    # Prediction horizon
```

### Trader Strategy Submission

The trader runs its strategy on the model assigned by the pairing (no
self-selected promotion pool).

```python
class L2StrategySubmission(bt.Synapse):
    role: str = "trader"
    strategy_id: str               # Unique strategy identifier
    model_ids_used: List[str]      # The validator-assigned researcher model(s)
    trading_mode: str              # "paper" | "live_capital"
    position_log: List[Dict]       # Signed position records
    realized_pnl: float            # Total P&L
    max_drawdown_pct: float        # Peak-to-trough loss
    omega_ratio: float             # Risk-adjusted return measure
    win_rate: float                # Fraction of profitable trades
```

### Pairing Synapses

```python
class PairAssignment(bt.Synapse):
    epoch_id: int; generation: int; pairing_seed: str
    researcher_uid: str; trader_uid: str   # chain-seeded, revealed at eval time

class PairScoreReport(bt.Synapse):
    model_composite: float; trading_composite: float; pair_composite: float
    pareto_rank: int; selection_score: float
    collusion_flagged: bool; miner_credit: float
```

### Public Feature Registry

Miners and validators share a published feature contract. Miners may use any subset:

```python
FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_30", "ret_60",       # Returns
    "vol_10", "vol_30", "vol_60",                           # Volatility
    "rsi_14", "rsi_30",                                     # Momentum
    "funding_rate", "funding_rate_ma_8h",                   # Funding
    "open_interest_change_1h", "open_interest_change_4h",   # Derivatives
    "volume_ratio_1h", "volume_ratio_4h",                   # Volume
    "ob_imbalance_top5", "ob_imbalance_top10",              # Order book
    "liquidation_intensity_1h",                             # Liquidations
    "taker_buy_sell_ratio_1h",                              # Flow
    "spread_bps", "price_range_pct_1h", "vwap_deviation",  # Microstructure
    "hour_of_day_sin", "hour_of_day_cos",                   # Temporal
    "day_of_week_sin", "day_of_week_cos",
]
```

---

## 5. Proprietary Boundaries

The following components are proprietary and NOT included in the open-source codebase:

| Component | Why Proprietary | What's Public Instead |
|-----------|----------------|----------------------|
| Validator benchmark dataset | Enterprise tick-by-tick data is the core competitive moat | Demo uses synthetic data with matching statistical properties |
| Overfitting detection metric | Proprietary evaluation tuned for GBDTs on financial time-series | Reference implementation using IS/OOS gap (see `scoring.py`) |
| Exact scoring weights | Subject to ongoing tuning | Published weight ranges and default values |
| Firm's deployment strategy | Prop trading operations | Architecture for buyback mechanism is documented |

Everything else — the scoring framework, synapse definitions, incentive design, anti-gaming mechanisms, and the pairing/genetic-selection logic — is fully transparent and open-source.

---

## 6. Differentiation from Existing Subnets

### vs. Taoshi / Vanta (SN8)

| Dimension | Taoshi (SN8) | Insignia |
|-----------|-------------|----------|
| Core output | Trading signals | ML model artifacts + validated strategies |
| Validation | Live performance only | Joint model + live validation per pair |
| Data source | Public prices | Researchers: proprietary tick benchmark; traders: real market |
| Simulation rigor | None | Full multi-metric backtest on institutional data |
| Specialization | Single role | Two roles (researcher + trader) matched genetically |
| Buyback mechanism | Not documented | Firm P&L → alpha token buybacks |

### Key Differentiators

1. **Paired desks**: Separate ML researchers and trading-operations engineers, matched and evaluated together like individuals in a genetic algorithm (similar to a pod shop)
2. **Genetic selection**: NSGA-II non-dominated sorting over `(model, strategy)` pairs, with marginal-contribution credit rewarding transferable skill
3. **Institutional Data**: Proprietary tick-by-tick benchmark creates ungameable model validation
4. **Short-Horizon Niche**: Minutes-to-hours prediction is technically distinct from medium-term signals

---

## 7. File Structure

```
subnet/
├── insignia/
│   ├── __init__.py           # Package definition
│   ├── protocol.py           # Synapses (model + trading + commit/reveal + pairing) and MinerRole
│   ├── scoring.py            # Composite scoring (7 model + 10 trading metrics) + combine_pair_scores
│   ├── pairing.py            # Paired genetic engine: chain-seeded pairing, NSGA-II matchmaker,
│   │                         #   marginal-contribution credit, collusion-graph detector
│   ├── incentive.py          # Anti-gaming mechanisms + commit-reveal + attack/defense matrix
│   └── cross_layer.py        # DEPRECATED (promotion/feedback replaced by pairing)
├── neurons/
│   ├── validator.py          # Unified PairedValidator (pairing → joint eval → NSGA-II → credit → set_weights)
│   ├── researcher_miner.py   # Role-aware researcher miner (ML model training)
│   ├── trader_miner.py       # Role-aware trader miner (paper/live trading on the assigned model)
│   ├── l1_miner.py / l1_validator.py / l2_miner.py / l2_validator.py  # legacy primitives reused by the above
├── tuning/
│   ├── attack_detector.py    # 28-vector detection incl. pairing vectors
│   ├── parameter_space.py    # Tuning space with the `pairing` parameter group
│   ├── optimizer.py          # OFFLINE NSGA-II mechanism tuner (distinct from the in-protocol GA)
│   ├── pc_vh_006_symbol_diversity.py # Symbol diversity enforcement policy
│   ├── sentinel_symbol_monitor.py # Symbol diversity monitoring and severity projection
│   ├── simulation.py         # Pair-based simulation harness with collusion/partner-gaming agents
│   └── orchestrator.py       # Tuning loop orchestration
├── testnet/
│   ├── config.py             # CommitRevealConfig, ValidationTimingConfig, ConsensusIntegrityConfig
│   ├── emulator.py           # Testnet emulator with commit-reveal support and route-diversity assignment
│   └── ...                   # Chain setup and wallet management
├── scripts/
│   └── run_demo.py           # Full paired-mechanism pipeline demonstration
├── docs/
│   ├── PAIRING_MECHANISM.md  # Authoritative single paired mechanism design
│   ├── INCENTIVE_MECHANISM.md # Incentive design + attack landscape + commit-reveal validation
│   ├── PARAMETER_TUNING_PLAN.md # Tuning strategy
│   ├── SUBNET_SPEC.md        # This document
│   └── TESTNET_DEPLOYMENT.md # Deployment guide with commit-reveal integration
├── program.md                # Agent swarm program (orchestration spec)
└── README.md                 # Quick start and overview
```
