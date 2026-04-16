# Insignia Subnet Specification

**Version**: 0.1.0
**Category**: Quantitative Finance / DeFi
**Track**: Bittensor Sovereign Infrastructure Hackathon

---

## 1. Subnet Identity

| Field | Value |
|-------|-------|
| **Name** | Insignia |
| **Tagline** | Decentralized Predictive Modeling for On-Chain Markets |
| **Architecture** | Two-Layer (Model Generation + Strategy Deployment) |
| **Primary Output** | Battle-tested ML model + trading strategy pairs |
| **Target Instruments** | BTC, ETH, SOL, AVAX, ADA perpetual futures |
| **Time Horizon** | Short-horizon (minutes to hours) |

---

## 2. Build Requirements Coverage

### Benchmark
**Target output quality metric miners are scored on.**

- **Layer 1**: Multi-metric evaluation vector (7 dimensions: Penalized F1, Penalized Sharpe, Max Drawdown, Variance Score, Overfitting Penalty, Feature Efficiency, Latency) scored against proprietary tick-by-tick benchmark dataset
- **Layer 2**: Real trading outcomes (10 dimensions: Realized P&L, Omega Ratio, Max Drawdown, Win Rate, Consistency, Model Attribution, Execution Quality, Annualized Volatility, Sharpe Ratio, Sortino Ratio) measured against actual market performance, with commit-reveal enforcement preventing post-hoc manipulation
- Published metric definitions with configurable weights (see `scoring.py`)

### Evaluation Loop
**How validators score miners; cost, scale consideration.**

- **L1**: Vectorized batch backtesting — O(n) per miner per epoch; parallelizable across miners
- **L2**: Incremental P&L tracking — O(1) per position close; continuous streaming
- Cost scales linearly with miner count, not quadratically
- Validators converge deterministically (same scorer + same data = same scores)

### Miner Task
**Specific, measurable, implementable task interface.**

- **L1 Task**: Train and submit ML model artifacts (ONNX/joblib) for directional prediction
  - Interface: `POST /l1/submit_model` with defined schema (see `protocol.py`)
  - Measurable via composite score vector
- **L2 Task**: Build and operate paper/live trading strategy using promoted L1 models
  - Interface: `POST /l2/submit_strategy` with position log
  - Measurable via realized P&L, Omega, drawdown, consistency, execution quality

### Incentive Design
**Why scoring rewards genuine quality; why top attack vectors fail.**

- Composite scoring prevents single-metric gaming
- Data asymmetry (miners: public data; validators: proprietary data) prevents overfitting to validation
- Cross-layer feedback rewards models that survive real deployment
- Commit-reveal scheme prevents post-hoc prediction manipulation and timing attacks (SHA-256 hashing with 128-bit nonces, commit window T-35s to T-5s, reveal window T+5s to T+20s)
- The core post-commit-reveal operating model is evaluated against 19 active surveillance vectors
- The latest orchestration run validated commit-reveal effectiveness at `0.76`, above the `0.667` acceptance floor, with a stronger operating margin than the prior run
- Sentinel classified the system as `SECURE_AND_IMPROVING`, with Sybil reduced to `0.195`, commitment violation reduced to `0.019`, and no new anomalies detected
- NSGA-II v13 R2 hit the primary target with breach_rate `3.5e-6`, honest_score `0.9795`, and separation `0.953`

### Market Demand
**Who pays for output and why.**

- **Primary buyer**: Subnet owner's prop trading firm (built-in demand floor)
- **Value proposition**: Model + strategy pairs with dual-layer performance proof (simulation + live)
- **Future buyers**: External quant funds seeking validated trading strategies

### Sovereignty Test
**Subnet survives if any single cloud, company, API disappears.**

- L1 miners run their own compute — no cloud dependency
- L1 validators can fall back to premium public data sources
- L2 paper trading can switch exchange price feeds trivially
- Core architecture (competitive model selection + live validation) survives any single provider outage

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INSIGNIA SUBNET                             │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────┐ │
│  │   LAYER 1        │    │   LAYER 2        │    │  OUTPUT   │ │
│  │                  │    │                  │    │           │ │
│  │  ⛏ Miners:       │    │  📈 Miners:       │    │  🏢 Firm  │ │
│  │  Train ML models ├───►│  Paper/live      ├───►│  deploys  │ │
│  │  (GBDT, Neural,  │    │  trading with    │    │  winning  │ │
│  │   Ensemble)      │    │  promoted models │    │  pairs    │ │
│  │                  │    │                  │    │           │ │
│  │  ✓ Validators:   │    │  ✓ Validators:   │    │  Buyback  │ │
│  │  Score on prop.  │    │  Score on real   │    │  → token  │ │
│  │  benchmark data  │    │  trading P&L     │    │  value    │ │
│  └────────┬─────────┘    └────────┬─────────┘    └───────────┘ │
│           │                       │                             │
│           └───────── Feedback ◄───┘                             │
│            L2 results adjust L1 scores                          │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Layer Data Flow

1. L1 miners submit model artifacts → L1 validators score on proprietary benchmark
2. Top-N models promoted to L2 pool (configurable, default N=10)
3. L2 miners build strategies around promoted models → paper/live trading
4. L2 validators score real trading outcomes
5. L2 performance feeds back to adjust L1 model scores retroactively
6. Subnet owner deploys winning model+strategy pairs

---

## 4. Technical Interfaces

### Layer 1 Model Submission

```python
class L1ModelSubmission(bt.Synapse):
    model_artifact: bytes          # Serialized model (ONNX/joblib, max 50MB)
    model_type: str                # gbdt | neural | ensemble | other
    features_used: List[str]       # From published feature registry
    training_window_start: str     # ISO datetime
    training_window_end: str       # ISO datetime
    hyperparams: Dict[str, Any]    # Full hyperparameter set
    target_instrument: str         # e.g., "BTC-USDT-PERP"
    target_horizon_minutes: int    # Prediction horizon
```

### Layer 2 Strategy Submission

```python
class L2StrategySubmission(bt.Synapse):
    strategy_id: str               # Unique strategy identifier
    model_ids_used: List[str]      # Which L1 models are inferenced
    trading_mode: str              # "paper" | "live_capital"
    position_log: List[Dict]       # Signed position records
    realized_pnl: float            # Total P&L
    max_drawdown_pct: float        # Peak-to-trough loss
    omega_ratio: float             # Risk-adjusted return measure
    win_rate: float                # Fraction of profitable trades
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

Everything else — the scoring framework, synapse definitions, incentive design, anti-gaming mechanisms, cross-layer logic — is fully transparent and open-source.

---

## 6. Differentiation from Existing Subnets

### vs. Taoshi / Vanta (SN8)

| Dimension | Taoshi (SN8) | Insignia |
|-----------|-------------|----------|
| Core output | Trading signals | ML model artifacts + validated strategies |
| Validation | Live performance only | Simulation + live chained together |
| Data source | Public prices | L1: proprietary tick data, L2: real market |
| Simulation rigor | None | Full multi-metric backtest on institutional data |
| Buyback mechanism | Not documented | Firm P&L → alpha token buybacks |

### Key Differentiators

1. **The Bridge**: First subnet to chain rigorous ML model competition with mandatory live validation
2. **Institutional Data**: Proprietary tick-by-tick benchmark creates ungameable validation
3. **Short-Horizon Niche**: Minutes-to-hours prediction is technically distinct from medium-term signals

---

## 7. File Structure

```
subnet/
├── insignia/
│   ├── __init__.py           # Package definition
│   ├── protocol.py           # Bittensor Synapse definitions (L1 + L2 + commit/reveal)
│   ├── scoring.py            # Composite scoring engine (L1: 7 metrics, L2: 10 metrics)
│   ├── incentive.py          # Anti-gaming mechanisms + commit-reveal + attack/defense matrix
│   └── cross_layer.py        # Model promotion + feedback loop engine
├── neurons/
│   ├── l1_miner.py           # Layer 1 miner template (model training + commit/reveal)
│   ├── l1_validator.py       # Layer 1 validator (evaluation + commitment verification)
│   ├── l2_miner.py           # Layer 2 miner template (paper trading + commit/reveal)
│   └── l2_validator.py       # Layer 2 validator (P&L tracking + scoring)
├── tuning/
│   ├── attack_detector.py    # 19-vector post-commit-reveal attack detection
│   ├── parameter_space.py    # 73-parameter tuning space with defense and routing parameters
│   ├── optimizer.py          # NSGA-II multi-objective optimization
│   ├── pc_vh_006_symbol_diversity.py # Symbol diversity enforcement policy
│   ├── sentinel_symbol_monitor.py # Symbol diversity monitoring and severity projection
│   ├── simulation.py         # Full pipeline simulation harness with stable MCP route assignment
│   └── orchestrator.py       # Tuning loop orchestration
├── testnet/
│   ├── config.py             # CommitRevealConfig, ValidationTimingConfig, ConsensusIntegrityConfig
│   ├── emulator.py           # Testnet emulator with commit-reveal support and route-diversity assignment
│   └── ...                   # Chain setup and wallet management
├── scripts/
│   └── run_demo.py           # Full end-to-end pipeline demonstration
├── docs/
│   ├── INCENTIVE_MECHANISM.md # Incentive design + attack landscape + commit-reveal validation
│   ├── PARAMETER_TUNING_PLAN.md # 60-parameter tuning strategy
│   ├── SUBNET_SPEC.md        # This document
│   └── TESTNET_DEPLOYMENT.md # Deployment guide with commit-reveal integration
├── program.md                # Agent swarm program (orchestration spec)
└── README.md                 # Quick start and overview
```
