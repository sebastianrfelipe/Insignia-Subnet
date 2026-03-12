# Insignia Subnet

**Decentralized Predictive Modeling for On-Chain Markets**

A Bittensor subnet proposal for the Sovereign Infrastructure Hackathon (March 11-15, 2026).

---

## What Is This?

Insignia is a two-layer Bittensor subnet that produces **battle-tested ML model + trading strategy pairs** for short-horizon crypto markets. It solves a critical limitation of existing quantitative subnets: the gap between simulation quality and real-world deployment performance.

**Layer 1 (Model Generation):** Distributed miners compete to train the best predictive ML models. Validators score submissions against a proprietary institutional-grade benchmark dataset that miners never see.

**Layer 2 (Strategy Deployment):** Top models from Layer 1 are promoted to Layer 2, where a second set of miners build live trading strategies around them. Validators score strategies on real market outcomes — actual P&L, drawdown, risk metrics.

**The result:** Model + strategy pairs with both algorithmic proof (Layer 1 simulation) and empirical proof (Layer 2 deployment). Neither layer alone produces this — the chain is the innovation.

---

## Quick Start

### Run the Full Pipeline Demo

```bash
cd subnet
pip install numpy pandas scikit-learn joblib
python3 scripts/run_demo.py
```

This runs the complete two-layer pipeline with synthetic data:
1. 8 L1 miners train and submit ML models
2. L1 validator evaluates all models across multiple market regimes
3. Top 4 models are promoted to the L2 pool
4. 4 L2 miners build paper trading strategies using promoted models
5. L2 validator scores strategies on simulated real-market outcomes
6. Cross-layer feedback adjusts L1 model scores based on L2 results

### Run Individual Components

```bash
python3 neurons/l1_miner.py      # L1 miner demo (train + submit)
python3 neurons/l1_validator.py   # L1 validator demo (evaluate 5 miners)
python3 neurons/l2_miner.py      # L2 miner demo (paper trading)
python3 neurons/l2_validator.py   # L2 validator demo (score strategies)
```

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌───────────┐
│   LAYER 1        │    │   LAYER 2        │    │  OUTPUT   │
│                  │    │                  │    │           │
│  Miners:         │    │  Miners:         │    │  Firm     │
│  Train ML models ├───>│  Paper/live      ├───>│  deploys  │
│  Submit to       │    │  trading with    │    │  winning  │
│  validators      │    │  promoted models │    │  pairs    │
│                  │    │                  │    │           │
│  Validators:     │    │  Validators:     │    │  Buyback  │
│  Score on prop.  │    │  Score on real   │    │  → token  │
│  benchmark data  │    │  trading P&L     │    │  value    │
└────────┬─────────┘    └────────┬─────────┘    └───────────┘
         │                       │
         └───── Feedback <───────┘
          L2 results adjust L1 scores
```

---

## Hackathon Build Requirements

| Requirement | Our Answer |
|---|---|
| **Benchmark** | L1: 7-metric evaluation vector (directional accuracy, Sharpe, drawdown, stability, overfitting, feature efficiency, latency). L2: 6-metric vector (P&L, Omega, drawdown, win rate, consistency, model attribution). |
| **Evaluation Loop** | L1: vectorized batch backtesting on proprietary data. L2: incremental P&L tracking with continuous position streaming. Both cost-efficient and scalable. |
| **Miner Task** | L1: Train and submit ML model artifacts via defined API. L2: Build and operate paper trading strategies using promoted models. Both with writable interfaces. |
| **Incentive Design** | Composite multi-metric scoring prevents gaming. Data asymmetry prevents overfitting. Cross-layer feedback rewards real deployment success. 9 attack vectors documented with defenses. |
| **Market Demand** | Subnet owner's prop firm is the built-in buyer. Model + strategy pairs with dual-layer proof are a stronger product than signal-only subnets. |
| **Sovereignty Test** | L1 miners: own compute. L1 validators: fallback to public data. L2 trading: switch exchange feeds. No single provider dependency. |

---

## What's Proprietary vs. What's Open

| Open (in this repo) | Proprietary (not included) |
|---|---|
| Full scoring framework (7 L1 + 6 L2 metrics) | Validator benchmark dataset (enterprise tick data) |
| All Synapse/protocol definitions | Production overfitting detector implementation |
| Anti-gaming mechanisms (fingerprinting, copy detection) | Exact scoring weight tuning |
| Cross-layer promotion + feedback logic | Firm deployment strategy |
| Paper trading engine with slippage model | |
| Complete end-to-end demo | |
| Incentive design + attack/defense analysis | |

The proprietary components are behind clean abstract interfaces (see `OverfittingDetector` in `scoring.py` and `BenchmarkDataProvider` in `l1_validator.py`). Judges can evaluate the architecture on design merit while understanding exactly where the competitive moat lies.

---

## Key Files

| File | Description |
|------|-------------|
| [`insignia/protocol.py`](insignia/protocol.py) | Bittensor Synapse definitions for both layers |
| [`insignia/scoring.py`](insignia/scoring.py) | Composite scoring engine with all metric implementations |
| [`insignia/incentive.py`](insignia/incentive.py) | Anti-gaming mechanisms + formal attack/defense matrix |
| [`insignia/cross_layer.py`](insignia/cross_layer.py) | Model promotion criteria + feedback loop engine |
| [`neurons/l1_miner.py`](neurons/l1_miner.py) | L1 miner template: model training pipeline + submission |
| [`neurons/l1_validator.py`](neurons/l1_validator.py) | L1 validator: evaluation against benchmark + weight setting |
| [`neurons/l2_miner.py`](neurons/l2_miner.py) | L2 miner template: paper trading engine + strategy execution |
| [`neurons/l2_validator.py`](neurons/l2_validator.py) | L2 validator: P&L tracking, scoring, cross-layer feedback |
| [`scripts/run_demo.py`](scripts/run_demo.py) | Full end-to-end pipeline demonstration |
| [`docs/INCENTIVE_MECHANISM.md`](docs/INCENTIVE_MECHANISM.md) | Detailed incentive design document |
| [`docs/SUBNET_SPEC.md`](docs/SUBNET_SPEC.md) | Complete subnet specification |

---

## Differentiation

**vs. Taoshi / Vanta (SN8):**
- Taoshi validates live trading signals only — no insight into model quality
- Insignia chains simulation + live validation — both required for deployment
- Proprietary tick data creates ungameable validation benchmark
- Buyback mechanism ties token value to real firm P&L

**vs. Simulation-only subnets:**
- Simulation subnets have no live performance proof
- Insignia requires Layer 2 live/paper validation before deployment eligibility

**The bridge between simulation and reality is the core innovation.**

---

## Post-Hackathon Roadmap

1. Register subnet on Bittensor mainnet
2. Connect proprietary enterprise tick data feed to L1 validators
3. Onboard L1 miner community with mining guides
4. Enable L2 paper trading on mainnet
5. Deploy first model+strategy pair from top L2 performer
6. Implement buyback mechanism (deployment P&L → token buybacks)
7. Phase 2: Extend to arbitrage and HFT model generation
8. Phase 3: External buyer API for validated strategy subscriptions

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.0.0
bittensor  # For production deployment
```

---

*Insignia Technologies — Sovereign Infrastructure Hackathon 2026*
