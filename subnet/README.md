# Insignia Subnet

**A decentralized network where independent participants compete to build the best predictive models — and prove they work in the real world.**

Built on [Bittensor](https://bittensor.com) for the Sovereign Infrastructure Hackathon (March 2026).

**Live Demo:** [output-sooty-ten.vercel.app](https://output-sooty-ten.vercel.app)

---

## What Is This?

Insignia is a two-layer competitive network for producing high-quality ML models and validating that they actually work when deployed.

**Layer 1 (Model Competition):** Participants ("miners") independently train ML models that predict future outcomes. Validators score every submission using a multi-metric system designed to be ungameable — accuracy alone isn't enough; consistency, risk, and generalization all matter.

**Layer 2 (Real-World Validation):** The best models from Layer 1 are promoted to Layer 2, where a second group of participants builds automated strategies around them. These strategies are scored on real outcomes — not simulations.

**Cross-Layer Feedback:** Layer 2 performance feeds back into Layer 1 scores. A model is only considered "good" if it actually works when deployed. This closes the gap between theory and practice.

---

## Demo Artifacts

Interactive visualizations showing every layer of the system:

| # | Artifact | What It Shows |
|---|----------|--------------|
| 1 | **Training Data Explorer** | 117 derivatives features, permutation importance, price action |
| 2 | **Model Performance** | Buy/sell signals on price, correct/wrong prediction blocks |
| 3 | **Scoring System** | 7-metric evaluation with per-metric breakdowns and attack context |
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
│  7-metric score  │    │  Score real P&L  │    │  → token  │
│  (mean−λ·std)    │    │  7 metrics       │    │  value    │
└────────┬─────────┘    └────────┬─────────┘    └───────────┘
         │                       │
         └───── Feedback <───────┘
          L2 results adjust L1 scores
```

---

## L1 Scoring (7 Metrics)

| Metric | Weight | Purpose |
|--------|--------|---------|
| Penalized F1 | 20% | Directional prediction quality with cross-regime consistency penalty |
| Penalized Sharpe Ratio | 20% | Risk-adjusted returns with variance penalty across sub-windows |
| Max Drawdown | 15% | Penalizes fragile models with large peak-to-trough losses |
| Variance Score | 15% | Cross-regime consistency — coefficient of variation across market regimes |
| Overfitting Penalty | 15% | Gap between in-sample and out-of-sample performance (proprietary metric) |
| Feature Efficiency | 5% | Penalizes models requiring excessive features |
| Latency | 10% | Inference speed — critical for short-horizon deployment |

All metrics use a **variance-penalized formulation** (`mean − λ·std`) across rolling windows, rewarding both performance and consistency.

---

## Project Structure

```
subnet/
├── insignia/               # Core protocol
│   ├── protocol.py         # Synapse definitions (L1 + L2)
│   ├── scoring.py          # 7-metric composite scorer
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
