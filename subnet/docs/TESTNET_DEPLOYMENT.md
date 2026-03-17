# Insignia Testnet Deployment & Emulator Guide

Complete guide for deploying the Insignia subnet on a Bittensor testnet and running the incentive-mechanism emulator for hyperparameter tuning.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start (Local Chain)](#quick-start-local-chain)
- [Manual Setup](#manual-setup)
- [Emulator Modes](#emulator-modes)
- [Public Testnet Deployment](#public-testnet-deployment)
- [Configuration Reference](#configuration-reference)
- [Monitoring & Dashboards](#monitoring--dashboards)
- [Troubleshooting](#troubleshooting)

---

## Overview

The testnet emulator bridges the Insignia simulation framework with a real Bittensor subtensor chain. This enables:

1. **Incentive mechanism validation** — Run simulated miners (honest + adversarial) against real Yuma consensus
2. **Hyperparameter tuning** — Evolutionary optimization (NSGA-II) of all 41 tunable parameters
3. **Attack resilience testing** — 9 documented attack vectors evaluated under realistic conditions
4. **Pre-mainnet rehearsal** — Full pipeline demo before deploying to mainnet

The emulator supports two chain targets:

| Target | Blocks | Cost | Use Case |
|--------|--------|------|----------|
| **Local** | 250ms (fast) | Free | Development, rapid iteration |
| **Testnet** | ~12s | Test TAO | Pre-mainnet validation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Insignia Emulator                         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Honest   │  │ Overfitter│  │ Copycat  │  │ Sybil    │   │
│  │ Miners   │  │ Miners   │  │ Miners   │  │ Miners   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       └──────────────┼──────────────┼──────────────┘         │
│                      ▼                                       │
│              ┌───────────────┐                               │
│              │  L1 Validator  │  ← Composite scoring         │
│              │  (7 metrics)   │  ← Anti-gaming checks        │
│              └───────┬───────┘  ← Fingerprinting             │
│                      │                                       │
│              ┌───────▼───────┐                               │
│              │   Promotion    │  → Top-N to Layer 2          │
│              │   Engine       │                               │
│              └───────┬───────┘                               │
│                      │                                       │
│  ┌──────────┐  ┌─────▼──────┐  ┌──────────┐                │
│  │ Honest   │  │ L2 Validator│  │ Copy     │                │
│  │ Traders  │──│ (6 metrics) │──│ Traders  │                │
│  └──────────┘  └─────┬──────┘  └──────────┘                │
│                      │                                       │
│              ┌───────▼───────┐                               │
│              │ Cross-Layer   │  ← L2→L1 feedback             │
│              │ Feedback      │  ← +15% / -10% adjustment     │
│              └───────┬───────┘                               │
│                      │                                       │
│              ┌───────▼───────┐                               │
│              │ set_weights() │  → Yuma Consensus             │
│              └───────┬───────┘                               │
└──────────────────────┼──────────────────────────────────────┘
                       │
               ┌───────▼───────┐
               │   Subtensor    │  (local or testnet)
               │   Chain        │
               └───────────────┘
```

---

## Prerequisites

### Software

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | `pyenv install 3.12` |
| Docker | ≥ 24.0 | [docker.com](https://docs.docker.com/get-docker/) |
| btcli | Latest | `pip install bittensor-cli` |
| bittensor SDK | Latest | `pip install bittensor` |

### Python Dependencies

```bash
cd subnet
pip install -r requirements.txt
pip install bittensor bittensor-cli pyyaml
```

---

## Quick Start (Local Chain)

The fastest way to get the emulator running on a local subtensor:

```bash
cd subnet

# Option 1: Automated setup script
bash testnet/setup_local.sh --mode single

# Option 2: Automated with evolutionary tuning
bash testnet/setup_local.sh --mode evolve --generations 10 --population 10

# Option 3: Manual step-by-step (see below)
```

### One-liner (no Docker, offline mode)

If you just want to run the emulator without a chain (offline simulation):

```bash
cd subnet
python -m testnet.run_emulator --mode single --network local --no-metrics
```

---

## Manual Setup

### Step 1: Start Local Subtensor

```bash
# Pull and start the local chain
docker run --rm -d \
  --name insignia-subtensor \
  -p 9944:9944 \
  -p 9945:9945 \
  -p 9933:9933 \
  ghcr.io/opentensor/subtensor-localnet:devnet-ready

# Verify
btcli subnet list --network ws://127.0.0.1:9945
```

Or use the provided Docker Compose:

```bash
cd subnet
docker compose -f testnet/docker-compose.testnet.yml up -d subtensor
```

### Step 2: Create Wallets

```bash
# Owner wallet (creates the subnet)
btcli wallet create --wallet.name insignia-owner --no-password

# Validator wallet
btcli wallet create --wallet.name insignia-validator --no-password

# Miner wallets (one per simulated agent)
for i in $(seq 0 7); do
  btcli wallet create --wallet.name "insignia-miner-${i}" --no-password
done
```

### Step 3: Fund Wallets (Local Chain)

On the local chain, use the pre-funded `alice` account:

```bash
# Get alice's address
btcli wallet overview --wallet.name alice --network ws://127.0.0.1:9945

# Transfer to each wallet
btcli wallet transfer \
  --wallet.name alice \
  --destination <OWNER_ADDRESS> \
  --amount 100000 \
  --network ws://127.0.0.1:9945 \
  --no-prompt
```

### Step 4: Create Subnet

```bash
btcli subnet create \
  --wallet.name insignia-owner \
  --network ws://127.0.0.1:9945 \
  --no-prompt

# Note the netuid from the output
btcli subnet list --network ws://127.0.0.1:9945
```

### Step 5: Register Neurons

```bash
NETUID=1  # Replace with your actual netuid

# Register validator
btcli subnets register \
  --netuid $NETUID \
  --wallet.name insignia-validator \
  --network ws://127.0.0.1:9945 \
  --no-prompt

# Register miners
for i in $(seq 0 7); do
  btcli subnets register \
    --netuid $NETUID \
    --wallet.name "insignia-miner-${i}" \
    --network ws://127.0.0.1:9945 \
    --no-prompt
done
```

### Step 6: Stake Validator

```bash
btcli stake add \
  --wallet.name insignia-validator \
  --amount 1000 \
  --network ws://127.0.0.1:9945 \
  --no-prompt
```

### Step 7: Run the Emulator

```bash
cd subnet

# Single epoch (quick test)
python -m testnet.run_emulator \
  --mode single \
  --network local \
  --netuid 1

# Evolutionary tuning
python -m testnet.run_emulator \
  --mode evolve \
  --network local \
  --netuid 1 \
  --generations 20 \
  --population 15 \
  --output testnet_results/
```

---

## Emulator Modes

### `single` — Quick Validation

Run one simulation epoch with default parameters. Useful for verifying the setup works end-to-end.

```bash
python -m testnet.run_emulator --mode single
```

### `sweep` — Parameter Sweep

Run N random parameter configurations and compare results. Good for initial exploration of the parameter space.

```bash
python -m testnet.run_emulator --mode sweep --n-configs 20
```

### `evolve` — Evolutionary Optimization

Run NSGA-II evolutionary optimization across generations. Each generation evaluates a population of parameter configurations, evolves the best, and iterates. This is the primary mode for serious hyperparameter tuning.

```bash
python -m testnet.run_emulator \
  --mode evolve \
  --generations 30 \
  --population 20 \
  --n-honest 8 \
  --n-adversarial 6
```

### `setup` — Chain Setup Only

Create wallets, subnet, register neurons, and fund wallets. No simulation.

```bash
python -m testnet.run_emulator --mode setup --network local
```

### `full` — Complete Pipeline

Run setup + evolutionary optimization + export in one command.

```bash
python -m testnet.run_emulator --mode full --network local --generations 20
```

### `status` — Health Check

Query current subnet state and wallet balances.

```bash
python -m testnet.run_emulator --mode status --network local --netuid 1
```

---

## Public Testnet Deployment

### Getting Test TAO

Test TAO is available via the Bittensor Discord faucet. Join the [Bittensor Discord](https://discord.gg/bittensor) and request tokens in `#faucet`.

### Connecting to Public Testnet

```bash
# Check testnet subnets
btcli subnet list --network test

# Create subnet on testnet (costs test TAO)
btcli subnet create --wallet.name insignia-owner --network test

# Run emulator against testnet
python -m testnet.run_emulator \
  --mode evolve \
  --network testnet \
  --netuid <YOUR_NETUID> \
  --generations 20
```

### EVM Testnet (MetaMask)

For EVM-compatible operations:

| Parameter | Value |
|-----------|-------|
| RPC URL | `https://test.chain.opentensor.ai` |
| Chain ID | `945` |
| Currency | TAO |

---

## Configuration Reference

### EmulatorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network` | `local` | Target: local, testnet, devnet |
| `netuid` | `None` | Existing subnet (auto-creates if None) |
| `n_l1_epochs` | `5` | L1 evaluation epochs per simulation |
| `n_l2_trading_steps` | `300` | L2 paper trading steps |
| `n_honest_l1` | `6` | Honest L1 miner agents |
| `n_adversarial_l1` | `4` | Adversarial L1 agents |
| `n_honest_l2` | `3` | Honest L2 trader agents |
| `n_adversarial_l2` | `1` | Adversarial L2 agents |

### Subnet Hyperparameters (On-Chain)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tempo` | `360` | Blocks between weight-setting epochs |
| `immunity_period` | `5000` | Blocks new neurons are immune |
| `max_validators` | `64` | Maximum validator count |
| `min_allowed_weights` | `1` | Minimum weights per validator |

### Insignia Scoring Parameters (41 Tunable)

See `tuning/parameter_space.py` for the full list. Key groups:

- **L1 Scoring Weights** (7 params): directional accuracy, Sharpe, drawdown, stability, overfitting, feature efficiency, latency
- **L2 Scoring Weights** (6 params): P&L, Omega ratio, drawdown, win rate, consistency, model attribution
- **Overfitting Detector** (2 params): gap threshold, decay rate
- **Promotion Criteria** (5 params): top-N, min epochs, max overfitting, decay limit, expiry
- **Anti-Gaming** (4 params): plagiarism threshold, copy-trade detection
- **Trading Engine** (6 params): slippage model, position limits, drawdown kill switch

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUBTENSOR_LOCAL_ENDPOINT` | Local chain WebSocket URL |
| `SUBTENSOR_TESTNET_ENDPOINT` | Public testnet WebSocket URL |
| `INSIGNIA_NETWORK` | Default network target |
| `INSIGNIA_NETUID` | Default subnet netuid |
| `INSIGNIA_OUTPUT_DIR` | Results output directory |

---

## Monitoring & Dashboards

### Start Monitoring Stack

```bash
cd subnet
docker compose -f testnet/docker-compose.testnet.yml up -d
```

### Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin/admin |
| Emulator Metrics | http://localhost:8001/metrics | — |

### Key Metrics

| Metric | Description |
|--------|-------------|
| `insignia_l1_composite_score` | L1 miner composite scores by agent type |
| `insignia_l2_composite_score` | L2 strategy scores |
| `insignia_attack_breach` | Attack breach flags (0/1) |
| `insignia_attack_severity` | Attack severity scores |
| `insignia_total_breaches` | Total active attack breaches |
| `insignia_best_fitness` | Best fitness per objective |
| `insignia_pareto_front_size` | Pareto front solution count |

---

## Troubleshooting

### btcli not found

```bash
pip install bittensor-cli
```

The emulator gracefully falls back to offline mode if btcli is unavailable.

### Cannot connect to local subtensor

```bash
# Check if container is running
docker ps | grep insignia-subtensor

# Check health
curl http://localhost:9933/health

# Restart
docker restart insignia-subtensor
```

### "Rate limited" errors in simulation

The emulator uses `force=True` to bypass rate limits during simulation epochs. If you see rate limit errors, ensure you're using the emulator's built-in simulation harness rather than calling validators directly.

### bittensor SDK import error

```bash
pip install bittensor
```

The emulator runs in offline mode if the SDK is not installed. All simulation and scoring logic works without the SDK — only on-chain weight-setting requires it.

### Out of memory during evolutionary tuning

Reduce population size and agent counts:

```bash
python -m testnet.run_emulator \
  --mode evolve \
  --generations 10 \
  --population 8 \
  --n-honest 4 \
  --n-adversarial 2
```

---

## Output Files

After a run, the emulator produces:

| File | Description |
|------|-------------|
| `emulator_results.json` | Full results with per-epoch scores, fitness, breach reports |
| `best_params.npy` | Best parameter vector (NumPy array) |
| `best_config.yaml` | Best configuration in human-readable YAML |
| `evolution_results.json` | Evolutionary tuning aggregate results |
| `sweep_results.json` | Parameter sweep aggregate results |

### Loading Results

```python
import numpy as np
import yaml
from tuning.parameter_space import decode, summarize_config

# Load best parameters
params = np.load("testnet_results/best_params.npy")
config = decode(params)
print(summarize_config(config))

# Load YAML config
with open("testnet_results/best_config.yaml") as f:
    yaml_config = yaml.safe_load(f)
```
