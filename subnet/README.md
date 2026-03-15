# Insignia Subnet

**Decentralized Predictive Modeling for On-Chain Markets**

A two-layer Bittensor subnet for the Sovereign Infrastructure Hackathon (March 2026).

**Live Demo:** [output-sooty-ten.vercel.app](https://output-sooty-ten.vercel.app)

---

## What Is This?

Insignia is a two-layer Bittensor subnet that produces **battle-tested ML model + trading strategy pairs** for short-horizon crypto markets.

**Layer 1 (Model Generation):** Miners compete to train the best predictive ML models. Validators score submissions against proprietary benchmark data using a 6-metric variance-penalized scoring system.

**Layer 2 (Strategy Deployment):** Top L1 models are promoted to L2, where miners build live trading strategies. Validators score real trading outcomes — P&L, drawdown, risk metrics.

**Cross-Layer Feedback:** L2 performance retroactively adjusts L1 scores, closing the simulation-to-reality gap.

---

## Demo Artifacts

Interactive visualizations showing every layer of the system:

| # | Artifact | What It Shows |
|---|----------|--------------|
| 1 | **Training Data Explorer** | 117 derivatives features, permutation importance, price action |
| 2 | **Model Performance** | Buy/sell signals on price, correct/wrong prediction blocks |
| 3 | **Scoring System** | 6-metric evaluation with per-metric breakdowns and attack context |
| 4 | **Network Flow** | 8-step flowchart from model training to emissions distribution |
| 5 | **Live Network Demo** | Two FastAPI servers (miner + validator) communicating over HTTP |

### View Static Demo

Open `output/index.html` in a browser, or visit the [Vercel deployment](https://output-sooty-ten.vercel.app).

### Build Visualizations from Data

```bash
cd subnet
uv sync
uv run python viz/build_viz.py
open viz/index.html
```

### Run Live Miner-Validator Demo

```bash
cd subnet
uv run python demo/run.py
# Open http://localhost:8000/dashboard
# Click "Run Epoch" to trigger evaluation cycles
```

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌───────────┐
│   LAYER 1        │    │   LAYER 2        │    │  OUTPUT   │
│                  │    │                  │    │           │
│  Miners:         │    │  Miners:         │    │  Firm     │
│  Train ML models ├───>│  Build trading   ├───>│  deploys  │
│  Submit to       │    │  strategies with │    │  winning  │
│  validators      │    │  promoted models │    │  pairs    │
│                  │    │                  │    │           │
│  Validators:     │    │  Validators:     │    │  Buyback  │
│  6-metric score  │    │  Score real P&L  │    │  → token  │
│  (mean−λ·std)    │    │  6 metrics       │    │  value    │
└────────┬─────────┘    └────────┬─────────┘    └───────────┘
         │                       │
         └───── Feedback <───────┘
          L2 results adjust L1 scores
```

---

## L1 Scoring (6 Metrics)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Penalized F1 | 25% | Directional quality + cross-regime consistency |
| Penalized Sharpe | 25% | Risk-adjusted returns + consistency |
| Max Drawdown | 15% | Penalizes fragile models with tail risk |
| Generalization Gap | 20% | Direct overfitting measure (train−test gap) |
| Feature Efficiency | 5% | Penalizes excessive feature dependencies |
| Latency | 10% | Inference speed for short-horizon trading |

All metrics use a **variance-penalized formulation** (`mean − λ·std`) across rolling windows, rewarding both performance and consistency.

---

## Project Structure

```
subnet/
├── insignia/               # Core protocol
│   ├── protocol.py         # Synapse definitions (L1 + L2)
│   ├── scoring.py          # 6-metric composite scorer
│   ├── incentive.py        # Anti-gaming mechanisms (9 attack defenses)
│   └── cross_layer.py      # Model promotion + feedback loop
├── neurons/                # Neuron implementations
│   ├── l1_miner.py         # L1 miner (model training)
│   ├── l1_validator.py     # L1 validator (evaluation + scoring)
│   ├── l2_miner.py         # L2 miner (paper trading)
│   └── l2_validator.py     # L2 validator (P&L scoring)
├── demo/                   # FastAPI miner-validator demo
│   ├── miner.py            # Miner server (:8001)
│   ├── validator.py        # Validator server (:8000) + dashboard
│   ├── models.py           # Pydantic request/response models
│   └── run.py              # Launch both servers
├── viz/                    # Visualization builders
│   ├── build_viz.py        # Entry point
│   ├── prepare.py          # Data loading + model caching
│   ├── build_input_data.py # Training data viz
│   ├── build_model_perf.py # Model performance viz
│   ├── build_scoring.py    # Scoring system viz
│   └── *.html / styles.css # Templates
├── tuning/                 # Parameter tuning framework
├── docs/                   # Specs and design docs
└── data/                   # Training data + model artifacts (gitignored)

output/                     # Static HTML/CSS/JS deployment (no Python needed)
```

---

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/):

```bash
cd subnet
uv sync
```

Core: `numpy`, `pandas`, `scikit-learn`, `joblib`
Demo: `fastapi`, `uvicorn`, `httpx`

---

*Insignia Technologies — Bittensor Sovereign Infrastructure Hackathon 2026*
