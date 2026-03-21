# Insignia Subnet — Agent Swarm Program

Autonomous agent swarm specification for deploying the Insignia subnet to
Bittensor testnet, simulating miners with OpenClaw agents, and continuously
tuning all L1, L2, and subnet hyperparameters via NSGA-II evolutionary
optimization with autoresearch-style experiment loops.

This file is the single source of truth the MCP server reads to orchestrate
the swarm. Agents do **not** modify this file — they execute it.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Agent Roles](#agent-roles)
3. [Phase 0: Environment Bootstrap](#phase-0-environment-bootstrap)
4. [Phase 1: Testnet Deployment](#phase-1-testnet-deployment)
5. [Phase 2: Miner Simulation](#phase-2-miner-simulation)
6. [Phase 3: Tuning Loop](#phase-3-tuning-loop)
7. [Phase 4: Attack Surveillance](#phase-4-attack-surveillance)
8. [Phase 5: Autoresearch Experiment Loop](#phase-5-autoresearch-experiment-loop)
9. [Convergence & Reset Protocol](#convergence--reset-protocol)
10. [Novel Attack Discovery Protocol](#novel-attack-discovery-protocol)
11. [MCP Server Interface](#mcp-server-interface)
12. [State Management](#state-management)
13. [Metrics & Observability](#metrics--observability)
14. [Constraints & Safety Rails](#constraints--safety-rails)

---

## System Overview

```
                        ┌────────────────────────┐
                        │     MCP SERVER          │
                        │  (Agent Orchestrator)   │
                        └────────┬───────────────┘
                                 │ dispatches tasks
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
    ┌──────▼──────┐     ┌───────▼───────┐     ┌──────▼──────┐
    │  DEPLOYER   │     │   SIMULATOR   │     │   TUNER     │
    │  AGENT      │     │   AGENT       │     │   AGENT     │
    │             │     │   (OpenClaw)  │     │   (NSGA-II) │
    └──────┬──────┘     └───────┬───────┘     └──────┬──────┘
           │                    │                     │
           │            ┌───────▼───────┐             │
           │            │   SENTINEL    │◄────────────┘
           │            │   AGENT       │  breach reports
           │            │   (Attack     │
           │            │    Detection) │
           └───────────►└───────┬───────┘
                                │
                        ┌───────▼───────┐
                        │  RESEARCHER   │
                        │  AGENT        │
                        │  (Autoresearch│
                        │   Loop)       │
                        └───────────────┘
```

The swarm operates as a continuous autonomous loop. Once started, it runs
indefinitely until manually stopped. Each agent has a distinct role but
they coordinate through shared state on disk and Prometheus metrics.

**Primary objective:** Find the parameter configuration across all 75+
dimensions (42 Insignia application-level + 33 Bittensor on-chain) that
maximizes honest miner composite scores while driving all 9 attack breach
rates to zero.

**Secondary objective:** Discover novel attack vectors not yet in the
attack detector, implement defenses, and re-tune accordingly.

---

## Agent Roles

### 1. DEPLOYER Agent

**Responsibility:** Infrastructure lifecycle — chain, wallets, subnet
registration, monitoring stack.

**Tools required:**
- Shell access (docker, btcli, python)
- File system read/write
- Network access to subtensor endpoints

**Capabilities:**
- Start/stop local subtensor or connect to public testnet
- Create and fund wallets
- Create subnet and register neurons
- Configure on-chain hyperparameters via `btcli subnets hyperparameters`
- Start/restart Prometheus + Grafana monitoring stack
- Health-check all infrastructure components

**State file:** `state/deployer_state.json`

### 2. SIMULATOR Agent (OpenClaw)

**Responsibility:** Run the full L1/L2 miner simulation pipeline with
the current parameter configuration. Produces `SimulationResult` objects
consumed by other agents.

**Tools required:**
- Python execution
- Read access to `tuning/`, `insignia/`, `neurons/`
- Write access to `state/` and `results/`

**Capabilities:**
- Instantiate all 8 agent types (Honest, Overfitter, Copycat, SingleMetricGamer, Sybil, Random, HonestTrader, CopyTrader)
- Run SimulationHarness with arbitrary parameter vectors
- Set weights on-chain via ChainInterface
- Export per-epoch results to JSON and Prometheus
- Support configurable agent populations and epoch counts

**State file:** `state/simulator_state.json`

### 3. TUNER Agent (NSGA-II + Autoresearch)

**Responsibility:** Multi-objective optimization of the full parameter
space. Proposes new parameter configurations, evaluates fitness, and
evolves the population.

**Tools required:**
- Python execution (pymoo, numpy)
- Read access to simulation results
- Write access to `results/`, `state/`

**Capabilities:**
- Run NSGA-II evolutionary optimization (pymoo)
- Maintain Pareto front of non-dominated solutions
- Select knee-point configurations for deployment
- Fall back to random search if pymoo unavailable
- Run autoresearch-style single-variable experiments
- Track experiment history in TSV format

**Optimization objectives (minimize all):**
1. `-mean_honest_score` — maximize honest miner performance
2. `breach_rate` — minimize fraction of attacks breached
3. `score_variance` — minimize variance across honest miners
4. `-score_separation` — maximize gap between honest and adversarial

**State file:** `state/tuner_state.json`

### 4. SENTINEL Agent (Attack Detection)

**Responsibility:** Continuous attack vector monitoring. Evaluates every
simulation result against all known attack vectors and triggers resets
when convergence toward an attack is detected.

**Tools required:**
- Python execution
- Read access to simulation results
- Write access to `state/sentinel_state.json`
- Ability to signal RESET to the orchestrator

**Capabilities:**
- Evaluate all 9 documented attack vectors per simulation
- Track breach trends across generations (moving averages)
- Detect convergence toward attack exploitation (3+ consecutive
  generations with increasing severity on any single attack)
- Trigger CONVERGENCE_ALERT when attack convergence detected
- Generate breach report summaries

**Attack vectors monitored:**
1. Overfitting exploitation
2. Model plagiarism
3. Single-metric gaming
4. Sybil attack
5. Copy-trading
6. Random baseline discrimination failure
7. Adversarial dominance
8. Insufficient honest/adversarial separation
9. Score concentration (HHI)

**State file:** `state/sentinel_state.json`

### 5. RESEARCHER Agent (Autoresearch Loop)

**Responsibility:** Karpathy-style autonomous experimentation. Instead
of evolutionary population search, this agent makes targeted single-
variable modifications to the parameter configuration, evaluates the
result, and keeps or discards — building understanding of the parameter
landscape through systematic ablation.

**Tools required:**
- Python execution
- Read/write access to `tuning/` code (can modify `autoresearch_loop.py`)
- Read access to simulation results
- Write access to `results/experiments.tsv`

**Capabilities:**
- Run time-budgeted experiments (configurable, default 5 min each)
- Modify one parameter group at a time, evaluate, keep/discard
- Maintain a results TSV with columns: `commit | breach_rate | honest_score | status | description`
- Never stop — runs indefinitely until interrupted
- Try increasingly radical changes when stuck
- Can propose new scoring formulas or attack detection heuristics

**The experiment loop:**
```
LOOP FOREVER:
  1. Read current best configuration from state/best_config.yaml
  2. Propose a modification (single param, param group, or structural)
  3. Run simulation with modified config
  4. Evaluate fitness (breach_rate, honest_score, separation)
  5. If improved → keep, update best_config.yaml
  6. If worse → discard, revert to previous config
  7. Log result to results/experiments.tsv
  8. If stuck for 10+ experiments → try more radical changes
  9. NEVER STOP. NEVER ASK FOR PERMISSION. RUN UNTIL KILLED.
```

**State file:** `state/researcher_state.json`

---

## Phase 0: Environment Bootstrap

The DEPLOYER agent executes this phase once at startup.

### Step 0.1: Verify Prerequisites

```bash
# Check Python
python3 --version  # >= 3.10

# Check Docker
docker --version  # >= 24.0

# Check btcli (install if missing)
btcli --version || pip install bittensor-cli

# Check bittensor SDK (install if missing)
python3 -c "import bittensor" || pip install bittensor

# Install subnet dependencies
cd subnet && pip install -r requirements.txt
pip install pymoo pyyaml prometheus_client
```

### Step 0.2: Create State Directories

```bash
mkdir -p state results testnet_results
```

### Step 0.3: Initialize State Files

Create `state/swarm_state.json`:
```json
{
  "phase": "bootstrap",
  "started_at": "<ISO_TIMESTAMP>",
  "generation": 0,
  "experiment_count": 0,
  "best_scalarized_fitness": null,
  "best_config_path": null,
  "active_alerts": [],
  "reset_count": 0,
  "novel_attacks_discovered": []
}
```

---

## Phase 1: Testnet Deployment

The DEPLOYER agent handles all chain infrastructure.

### Step 1.1: Determine Target Network

Read `INSIGNIA_NETWORK` environment variable. Defaults to `local`.

| Target | When to Use |
|--------|-------------|
| `local` | Development, rapid iteration (250ms blocks, free) |
| `testnet` | Pre-mainnet validation (12s blocks, test TAO) |

### Step 1.2: Start Chain (Local Mode)

```bash
docker compose -f testnet/docker-compose.testnet.yml up -d subtensor

# Wait for chain to be ready (poll health endpoint)
for i in $(seq 1 30); do
  curl -s http://localhost:9933/health && break
  sleep 2
done
```

### Step 1.3: Create Wallets

```bash
NETWORK_FLAG="ws://127.0.0.1:9945"  # or "test" for public testnet

# Owner wallet
btcli wallet create --wallet.name insignia-owner --no-password

# Validator wallet
btcli wallet create --wallet.name insignia-validator --no-password

# Miner wallets (one per simulated agent slot)
for i in $(seq 0 11); do
  btcli wallet create --wallet.name "insignia-miner-${i}" --no-password
done
```

### Step 1.4: Fund Wallets (Local Only)

```bash
# Transfer from pre-funded alice account
for WALLET in insignia-owner insignia-validator; do
  btcli wallet transfer \
    --wallet.name alice \
    --destination $(btcli wallet overview --wallet.name $WALLET --network $NETWORK_FLAG 2>/dev/null | grep -oP '5\w+' | head -1) \
    --amount 100000 \
    --network $NETWORK_FLAG \
    --no-prompt
done
```

For public testnet: request test TAO from the Bittensor Discord faucet.

### Step 1.5: Create Subnet

```bash
btcli subnet create \
  --wallet.name insignia-owner \
  --network $NETWORK_FLAG \
  --no-prompt

# Capture netuid
NETUID=$(btcli subnet list --network $NETWORK_FLAG 2>/dev/null | tail -n +2 | awk '{print $1}' | tail -1)
echo "NETUID=$NETUID" >> state/deployer_state.env
```

### Step 1.6: Register Neurons

```bash
# Validator
btcli subnets register \
  --netuid $NETUID \
  --wallet.name insignia-validator \
  --network $NETWORK_FLAG \
  --no-prompt

# Miners
for i in $(seq 0 11); do
  btcli subnets register \
    --netuid $NETUID \
    --wallet.name "insignia-miner-${i}" \
    --network $NETWORK_FLAG \
    --no-prompt
done

# Stake validator
btcli stake add \
  --wallet.name insignia-validator \
  --amount 1000 \
  --network $NETWORK_FLAG \
  --no-prompt
```

### Step 1.7: Configure Initial Subnet Hyperparameters

```bash
# Set the 6 most operationally relevant on-chain hyperparameters
btcli subnets hyperparameters set \
  --netuid $NETUID \
  --wallet.name insignia-owner \
  --param tempo --value 360 \
  --network $NETWORK_FLAG --no-prompt

btcli subnets hyperparameters set \
  --netuid $NETUID \
  --wallet.name insignia-owner \
  --param immunity_period --value 5000 \
  --network $NETWORK_FLAG --no-prompt

btcli subnets hyperparameters set \
  --netuid $NETUID \
  --wallet.name insignia-owner \
  --param min_allowed_weights --value 1 \
  --network $NETWORK_FLAG --no-prompt
```

### Step 1.8: Start Monitoring Stack

```bash
docker compose -f testnet/docker-compose.testnet.yml up -d prometheus grafana

# Verify
curl -s http://localhost:9090/-/ready  # Prometheus
curl -s http://localhost:3000/api/health  # Grafana
```

### Step 1.9: Update State

Write to `state/deployer_state.json`:
```json
{
  "network": "local",
  "endpoint": "ws://127.0.0.1:9945",
  "netuid": 1,
  "chain_healthy": true,
  "monitoring_healthy": true,
  "wallets_funded": true,
  "deployed_at": "<ISO_TIMESTAMP>"
}
```

Update `state/swarm_state.json`: `"phase": "deployed"`

---

## Phase 2: Miner Simulation

The SIMULATOR agent manages the simulation population.

### Step 2.1: Configure Agent Population

Default population mix for testnet simulation:

| Agent Type | Count | Layer | Purpose |
|------------|-------|-------|---------|
| HonestMiner | 6 | L1 | Baseline good-faith miners |
| OverfittingMiner | 2 | L1 | Test overfitting detection |
| CopycatMiner | 1 | L1 | Test plagiarism detection |
| SingleMetricGamer | 1 | L1 | Test composite scoring |
| SybilMiner | 2 | L1 | Test sybil detection |
| RandomMiner | 1 | L1 | Noise floor baseline |
| HonestTrader | 3 | L2 | Baseline trading strategies |
| CopyTrader | 1 | L2 | Test copy-trade detection |

Population is configurable. For heavier attack testing, scale adversarial
agent counts up to 2x the honest count.

### Step 2.2: Run Initial Baseline Simulation

```bash
cd subnet
python -m testnet.run_emulator \
  --mode single \
  --network local \
  --netuid $NETUID \
  --n-honest 6 \
  --n-adversarial 4 \
  --no-metrics
```

Record baseline fitness in `results/experiments.tsv`:
```
config_hash	breach_rate	honest_score	separation	status	description
baseline	0.4444	0.5234	0.0312	keep	default parameters baseline
```

### Step 2.3: Continuous Simulation Execution

The SIMULATOR runs simulations on demand from the TUNER and RESEARCHER
agents. Each simulation:

1. Receives a parameter vector (41-dimensional `np.ndarray`)
2. Decodes it via `tuning.parameter_space.decode()`
3. Creates the agent population via `create_default_agents()`
4. Runs `SimulationHarness.run(param_vector)`
5. Returns `SimulationResult` to the requesting agent
6. If netuid is configured, sets weights on-chain via `ChainInterface`
7. Exports metrics to Prometheus

---

## Phase 3: Tuning Loop

The TUNER agent runs the core optimization.

### Step 3.1: NSGA-II Evolutionary Optimization

Primary tuning mode. Uses pymoo's NSGA-II to search the 41-dimensional
Insignia parameter space.

```bash
cd subnet
python -m tuning.orchestrator \
  --mode optimize \
  --generations 50 \
  --population 30 \
  --n-honest 6 \
  --n-adversarial 2 \
  --n-epochs 3 \
  --n-steps 200 \
  --output results/ \
  --seed 42
```

**Configuration:**
- Population size: 30 (each individual is a full simulation)
- Generations: 50 (extendable if improving)
- Crossover: SBX (prob=0.9, eta=15)
- Mutation: Polynomial (eta=20)
- Repair: Weight normalization (L1 + L2 weights sum to 1.0)

**After each generation:**
1. Log Pareto front size, best fitness, diversity
2. Export to Prometheus
3. Check SENTINEL for convergence alerts
4. If alert → trigger RESET (see Convergence Protocol)
5. Save checkpoint to `results/gen_{N}_checkpoint.npy`

### Step 3.2: Subnet Hyperparameter Tuning

The 33 Bittensor on-chain parameters are tuned separately because they
require `btcli` calls and affect the chain globally (not per-simulation).

**Strategy:** Grid search over the 6 most impactful on-chain parameters,
evaluated by running the full NSGA-II inner loop per configuration:

| Parameter | Search Range | Steps |
|-----------|-------------|-------|
| `tempo` | [100, 200, 360, 720] | 4 |
| `immunity_period` | [1000, 3000, 5000, 10000] | 4 |
| `bonds_moving_avg` | [500000, 750000, 900000] | 3 |
| `adjustment_alpha` | [0, 32768, 58982] | 3 |
| `max_weight_limit` | [32768, 49152, 65535] | 3 |
| `min_allowed_weights` | [1, 4, 8] | 3 |

Each on-chain configuration is set via `btcli`, then the NSGA-II inner
loop runs 10 generations to evaluate its effect on the optimal Insignia
parameter set. The outer loop selects the on-chain configuration that
produces the best inner-loop Pareto front.

### Step 3.3: Joint Optimization Schedule

```
OUTER LOOP (on-chain hyperparameters):
  For each candidate on-chain config:
    1. DEPLOYER sets on-chain hyperparameters via btcli
    2. Wait for chain to propagate (tempo * 2 blocks)
    3. INNER LOOP (Insignia parameters):
       Run NSGA-II for N generations
       Record best Pareto front
    4. Compare to previous best outer config
    5. Keep if Pareto front dominates
```

---

## Phase 4: Attack Surveillance

The SENTINEL agent monitors every simulation result.

### Step 4.1: Per-Simulation Evaluation

After every simulation, SENTINEL:

1. Runs `AttackDetector.evaluate(sim_result)` → `BreachReport`
2. Records breach/severity for each of the 9 attack vectors
3. Updates rolling 5-generation moving averages per attack
4. Checks convergence conditions (see below)

### Step 4.2: Convergence Detection

An attack is **converging** if:
- Its severity has increased for 3+ consecutive generations, AND
- Current severity > 0.5, AND
- The attack is breached (boolean True)

A **convergence alert** triggers when ANY attack is converging.

### Step 4.3: Alert Escalation

| Alert Level | Condition | Action |
|-------------|-----------|--------|
| `INFO` | Any attack breached but severity < 0.3 | Log, continue |
| `WARNING` | Any attack breached with severity > 0.5 | Log, bias NSGA-II toward breach_rate objective |
| `CRITICAL` | Convergence detected (3+ gen increasing) | Trigger RESET |
| `EMERGENCY` | 5+ attacks breached simultaneously | Trigger FULL RESET + parameter space expansion |

### Step 4.4: Breach Trend Tracking

Maintain a per-attack time series in `state/sentinel_state.json`:
```json
{
  "attack_trends": {
    "overfitting_exploitation": {
      "history": [0.0, 0.1, 0.2, 0.35, 0.5],
      "moving_avg": 0.23,
      "consecutive_increases": 4,
      "converging": true
    },
    ...
  },
  "active_alerts": ["CRITICAL:overfitting_exploitation"],
  "total_resets": 2
}
```

---

## Phase 5: Autoresearch Experiment Loop

The RESEARCHER agent operates in parallel with NSGA-II, using Karpathy's
autoresearch methodology adapted for subnet parameter tuning.

### Core Philosophy

Where NSGA-II searches the parameter space as a population, the
RESEARCHER agent operates as a single-threaded scientist:

1. **Hypothesis:** "Increasing the overfitting penalty weight will reduce
   overfitting breach severity."
2. **Experiment:** Modify one parameter, run simulation, measure.
3. **Result:** Keep if improved, discard if not.
4. **Insight:** Build understanding of which parameters matter most.

This is complementary to NSGA-II — the RESEARCHER explores qualitative
insights and structural changes that evolutionary search cannot discover
(e.g., new scoring formulas, new agent types, new attack detectors).

### Experiment Types (Ordered by Risk)

| Type | Risk | Description |
|------|------|-------------|
| **Single-param nudge** | Low | Change one parameter by ±10-20% |
| **Group adjustment** | Low | Adjust all weights in one group proportionally |
| **Threshold sweep** | Medium | Sweep one threshold across its full range |
| **Structural** | Medium | Change scoring formula (e.g., add new metric) |
| **Agent modification** | High | Add new adversarial agent type |
| **Detector enhancement** | High | Add new attack detection heuristic |

### The Experiment Loop

```python
# Pseudocode for the RESEARCHER agent
LOOP FOREVER:
    best = load("state/best_config.yaml")
    idea = generate_experiment_idea(history, best)

    modified_config = apply_modification(best, idea)
    result = run_simulation(modified_config)
    fitness = evaluate_fitness(result)

    if fitness_improved(fitness, best_fitness):
        save("state/best_config.yaml", modified_config)
        log_result(status="keep", description=idea)
        best_fitness = fitness
    else:
        log_result(status="discard", description=idea)

    # If stuck for 10+ experiments with no improvement:
    if consecutive_discards > 10:
        idea = generate_radical_idea(history)
```

### Experiment Idea Generation

The RESEARCHER agent generates ideas using this priority system:

1. **Parameter sensitivity:** Start with the highest-impact parameters
   (L1/L2 weights, overfitting thresholds, feedback weights)
2. **Breach-driven:** If a specific attack is breached, target the
   parameters most likely to address it
3. **Ablation:** Remove or zero-out parameters to test if they matter
4. **Interaction effects:** After single-param experiments, test
   2-parameter combinations that showed individual promise
5. **Structural changes:** Modify scoring formulas, add metrics, change
   agent behaviors
6. **Radical changes:** If stuck, try extreme parameter values, invert
   assumptions, or redesign components

### Results Logging

All experiments are logged to `results/experiments.tsv`:

```
config_hash	breach_rate	honest_score	separation	variance	status	description
a1b2c3d	0.3333	0.5500	0.0450	0.0012	keep	increase overfitting_penalty weight to 0.25
b2c3d4e	0.4444	0.5200	0.0300	0.0015	discard	decrease feedback_bonus_weight to 0.05
c3d4e5f	0.2222	0.5800	0.0600	0.0010	keep	increase fingerprint_correlation_threshold to 0.98
d4e5f6g	0.2222	0.5900	0.0650	0.0009	keep	add variance penalty to L2 omega scoring
```

### Integration with NSGA-II

When the RESEARCHER finds an improvement:
1. Update `state/best_config.yaml`
2. Signal TUNER to seed next NSGA-II generation with the new config
3. TUNER injects the researcher's best config into the population as an
   elite individual, ensuring good discoveries propagate

---

## Convergence & Reset Protocol

When SENTINEL detects convergence toward an attack vector, the swarm
executes a coordinated reset.

### Reset Levels

#### Level 1: SOFT RESET (breach_rate warning)

1. TUNER increases the `breach_rate` objective weight by 2x in the
   scalarized fitness function
2. TUNER injects 5 random individuals into the population to increase
   diversity
3. RESEARCHER shifts focus to parameters related to the breached attack
4. No infrastructure restart required

#### Level 2: HARD RESET (convergence detected)

1. **STOP** all running simulations
2. TUNER saves current Pareto front to `results/pre_reset_{N}/`
3. TUNER reinitializes population:
   - 30% from current Pareto front (elite preservation)
   - 30% from the researcher's best configurations
   - 40% random (diversity injection)
4. SENTINEL resets convergence counters
5. TUNER restarts NSGA-II from generation 0 with the new population
6. Log reset event to `state/swarm_state.json`

#### Level 3: FULL RESET (5+ simultaneous breaches)

1. **STOP** everything
2. Save all state to `results/emergency_reset_{N}/`
3. DEPLOYER tears down and recreates the subnet:
   - Stop chain container
   - Clear wallet state
   - Restart chain
   - Re-create subnet with modified on-chain hyperparameters
4. TUNER reinitializes with pure random population
5. RESEARCHER starts with clean experiment history
6. Resume from Phase 1

### Reset State Machine

```
RUNNING → [WARNING] → SOFT_RESET → RUNNING
RUNNING → [CRITICAL] → HARD_RESET → RUNNING
RUNNING → [EMERGENCY] → FULL_RESET → Phase 1
```

### Post-Reset Tuning

After any reset, the TUNER adjusts its strategy:

1. **Increase breach_rate weight** in scalarized fitness:
   `breach_weight = base_weight * (1 + 0.5 * reset_count)`
2. **Tighten attack thresholds** in SENTINEL by 10% per reset
3. **Expand parameter ranges** for parameters related to breached attacks
4. **Log root cause analysis** of what parameter drift caused convergence

---

## Novel Attack Discovery Protocol

When the RESEARCHER or SENTINEL identifies behavior that doesn't match
any of the 9 documented attack vectors but appears exploitative:

### Detection Signals

- A non-honest agent type achieves top-3 ranking with a novel strategy
- Score distribution develops unexpected modality
- Cross-layer feedback creates an amplification loop
- Emission concentration (HHI) increases without any documented attack
  being breached

### Response Protocol

1. **SENTINEL** flags the anomaly with `NOVEL_ATTACK_SUSPECTED` alert
2. **RESEARCHER** investigates:
   a. Run targeted simulations isolating the suspected behavior
   b. Identify which parameter configuration enables the exploit
   c. Characterize the attack (document breach condition + severity metric)
3. **RESEARCHER** implements a defense:
   a. Add new `_check_*` method to `AttackDetector` class
   b. Add new breach type to `BreachReport`
   c. Run validation simulations confirming the defense works
4. **TUNER** re-runs NSGA-II with the expanded attack surface
5. **Log** the new attack to `state/swarm_state.json` under
   `novel_attacks_discovered`

### Implementation Template

When adding a new attack check to `tuning/attack_detector.py`:

```python
def _check_new_attack(self, result: SimulationResult) -> AttackBreach:
    """
    Check for <ATTACK_NAME>.

    Breach condition: <DESCRIBE WHEN THIS IS BREACHED>
    """
    # Extract relevant scores/metrics from result
    # Compute breach condition and severity
    # Return AttackBreach(name, breached, severity, description)
    ...
```

Then add to `AttackDetector.evaluate()`:
```python
report.breaches.append(self._check_new_attack(result))
```

---

## MCP Server Interface

The MCP server orchestrates the agent swarm. Each agent registers its
capabilities and the MCP server dispatches tasks.

### Agent Registration

Each agent exposes these MCP tools:

```json
{
  "deployer": {
    "tools": [
      "deploy_chain",
      "create_wallets",
      "create_subnet",
      "register_neurons",
      "set_hyperparameters",
      "health_check",
      "teardown",
      "restart"
    ]
  },
  "simulator": {
    "tools": [
      "run_simulation",
      "run_batch_simulations",
      "set_weights_on_chain",
      "get_simulation_result"
    ]
  },
  "tuner": {
    "tools": [
      "start_nsga2",
      "stop_nsga2",
      "get_pareto_front",
      "get_best_config",
      "inject_elite",
      "reset_population",
      "set_objective_weights"
    ]
  },
  "sentinel": {
    "tools": [
      "evaluate_attacks",
      "get_breach_report",
      "get_convergence_status",
      "reset_counters",
      "check_alerts"
    ]
  },
  "researcher": {
    "tools": [
      "run_experiment",
      "get_experiment_history",
      "propose_modification",
      "apply_structural_change"
    ]
  }
}
```

### MCP Orchestration Loop

```
LOOP FOREVER:
  1. DEPLOYER.health_check()
     → If unhealthy: DEPLOYER.restart()

  2. TUNER.get_best_config() → current_best
     SIMULATOR.run_simulation(current_best) → sim_result

  3. SENTINEL.evaluate_attacks(sim_result) → breach_report
     → If CRITICAL: execute RESET protocol
     → If WARNING: TUNER.set_objective_weights(increased_breach_weight)

  4. TUNER.start_nsga2(generation_config)
     → Each individual calls SIMULATOR.run_simulation()
     → Each result calls SENTINEL.evaluate_attacks()
     → TUNER evolves population

  5. RESEARCHER.run_experiment()
     → If improvement: TUNER.inject_elite(researcher_best)

  6. Every 10 generations:
     DEPLOYER.set_hyperparameters(outer_loop_candidate)

  7. Export state, metrics, checkpoints
     Sleep until next cycle
```

### MCP Message Format

Inter-agent messages use this schema:

```json
{
  "from": "sentinel",
  "to": "orchestrator",
  "type": "CONVERGENCE_ALERT",
  "severity": "CRITICAL",
  "payload": {
    "attack": "overfitting_exploitation",
    "severity_history": [0.3, 0.4, 0.5, 0.6],
    "consecutive_increases": 4,
    "recommended_action": "HARD_RESET"
  },
  "timestamp": "<ISO_TIMESTAMP>"
}
```

---

## State Management

All state is persisted to disk under `subnet/state/`.

### State Files

| File | Owner | Contents |
|------|-------|----------|
| `swarm_state.json` | Orchestrator | Global phase, generation, alerts, resets |
| `deployer_state.json` | DEPLOYER | Network config, netuid, health status |
| `simulator_state.json` | SIMULATOR | Current agent population, last result |
| `tuner_state.json` | TUNER | NSGA-II generation, Pareto front, population |
| `sentinel_state.json` | SENTINEL | Attack trends, alerts, convergence flags |
| `researcher_state.json` | RESEARCHER | Experiment history, current hypothesis |
| `best_config.yaml` | Shared | Current best parameter configuration |

### State Transitions

```
bootstrap → deployed → tuning → [soft_reset | hard_reset | full_reset] → tuning
                                                                              ↑
                                                                              └── (indefinite loop)
```

### Checkpointing

Every 5 generations, save a full checkpoint:
```
results/checkpoint_gen_{N}/
  ├── pareto_front.npy
  ├── pareto_X.npy
  ├── best_config.yaml
  ├── best_params.npy
  ├── swarm_state.json
  ├── breach_history.json
  └── experiments.tsv
```

---

## Metrics & Observability

### Prometheus Metrics (port 8001)

**Simulation metrics:**
- `insignia_l1_composite_score{agent_type, uid}` — per-miner L1 scores
- `insignia_l2_composite_score{agent_type, uid}` — per-strategy L2 scores
- `insignia_honest_mean_score` — mean honest miner score
- `insignia_adversarial_mean_score` — mean adversarial miner score
- `insignia_score_separation` — honest - adversarial gap

**Attack metrics:**
- `insignia_attack_breach{attack_name}` — 0/1 per attack
- `insignia_attack_severity{attack_name}` — 0.0-1.0 per attack
- `insignia_total_breaches` — count of active breaches
- `insignia_convergence_alert{attack_name}` — 0/1 convergence flag

**Optimizer metrics:**
- `insignia_generation` — current NSGA-II generation
- `insignia_best_fitness{objective}` — best fitness per objective
- `insignia_pareto_front_size` — Pareto front solution count
- `insignia_population_diversity` — mean parameter std across population

**Researcher metrics:**
- `insignia_experiment_count` — total experiments run
- `insignia_experiment_keep_rate` — fraction of experiments kept
- `insignia_consecutive_discards` — current discard streak

**Infrastructure metrics:**
- `insignia_chain_block` — current chain block number
- `insignia_chain_healthy` — 0/1 chain health
- `insignia_simulation_duration_seconds` — simulation wall time

### Grafana Dashboards

Pre-configured at `monitoring/grafana/dashboards/insignia-tuning.json`:

1. **Overview Panel** — Generation, best fitness, breach count, uptime
2. **Scoring Distribution** — L1/L2 score histograms by agent type
3. **Attack Status** — Traffic light for each of 9 attacks
4. **Convergence Monitor** — Per-attack severity time series with trend lines
5. **Pareto Front** — Scatter plot of multi-objective fitness
6. **Researcher Progress** — Experiment keep/discard rate over time
7. **Parameter Evolution** — How top parameters change across generations

---

## Constraints & Safety Rails

### Hard Constraints (Never Violate)

1. **L1 weights must sum to 1.0** — enforced by `repair_weights()`
2. **L2 weights must sum to 1.0** — enforced by `repair_weights()`
3. **All parameters within defined bounds** — enforced by `np.clip()`
4. **Maximum 1 subnet on testnet** — avoid wasting test TAO
5. **Never modify `insignia/protocol.py`** — protocol is frozen
6. **Never modify scoring metric formulas in production** without
   RESEARCHER validation (minimum 10 experiment confirmations)

### Soft Constraints (Prefer but Allow Override)

1. **Prefer simpler configurations** — fewer non-default parameters
2. **Prefer Pareto-optimal solutions** — don't pick dominated configs
3. **Prefer configurations robust to seed variation** — test with 3+
   random seeds before accepting as "best"
4. **Prefer configurations that defend ALL attacks** — zero-breach is
   the target, not just low breach rate

### Safety Rails

1. **Experiment timeout:** Each simulation must complete within 10
   minutes. Kill and discard if exceeded.
2. **Resource limits:** Maximum 4 concurrent simulations. Queue others.
3. **Disk space:** Alert if `results/` exceeds 10 GB. Archive old
   checkpoints.
4. **State backup:** Copy `state/` to `state_backup/` before any RESET.
5. **Graceful shutdown:** On SIGTERM, save all state, complete current
   simulation, then exit.

---

## Appendix A: Parameter Quick Reference

### Insignia Application Parameters (41)

| # | Parameter | Range | Group |
|---|-----------|-------|-------|
| 1 | `l1_penalized_f1` | [0.05, 0.40] | L1 weights |
| 2 | `l1_penalized_sharpe` | [0.05, 0.40] | L1 weights |
| 3 | `l1_max_drawdown` | [0.05, 0.30] | L1 weights |
| 4 | `l1_variance_score` | [0.05, 0.30] | L1 weights |
| 5 | `l1_overfitting_penalty` | [0.05, 0.35] | L1 weights |
| 6 | `l1_feature_efficiency` | [0.01, 0.15] | L1 weights |
| 7 | `l1_latency` | [0.01, 0.20] | L1 weights |
| 8 | `l2_realized_pnl` | [0.05, 0.40] | L2 weights |
| 9 | `l2_omega` | [0.05, 0.30] | L2 weights |
| 10 | `l2_max_drawdown` | [0.05, 0.30] | L2 weights |
| 11 | `l2_win_rate` | [0.05, 0.25] | L2 weights |
| 12 | `l2_consistency` | [0.05, 0.30] | L2 weights |
| 13 | `l2_model_attribution` | [0.01, 0.25] | L2 weights |
| 14 | `l2_execution_quality` | [0.05, 0.30] | L2 weights |
| 15 | `overfit_gap_threshold` | [0.05, 0.40] | Overfitting |
| 16 | `overfit_decay_rate` | [1.0, 15.0] | Overfitting |
| 17 | `promotion_top_n` | [3, 20] | Promotion |
| 18 | `promotion_min_consecutive_epochs` | [1, 5] | Promotion |
| 19 | `promotion_max_overfitting_score` | [0.1, 0.6] | Promotion |
| 20 | `promotion_max_score_decay_pct` | [0.05, 0.4] | Promotion |
| 21 | `promotion_expiry_epochs` | [3, 15] | Promotion |
| 22 | `feedback_bonus_weight` | [0.0, 0.40] | Feedback |
| 23 | `feedback_penalty_weight` | [0.0, 0.30] | Feedback |
| 24 | `fingerprint_correlation_threshold` | [0.80, 0.99] | Anti-gaming |
| 25 | `copy_trade_time_tolerance` | [10, 300] | Anti-gaming |
| 26 | `copy_trade_size_tolerance` | [0.01, 0.15] | Anti-gaming |
| 27 | `copy_trade_correlation_threshold` | [0.75, 0.98] | Anti-gaming |
| 28 | `slippage_base_spread_bps` | [0.5, 10.0] | Trading |
| 29 | `slippage_vol_impact_factor` | [0.1, 2.0] | Trading |
| 30 | `slippage_size_impact_factor` | [0.01, 0.5] | Trading |
| 31 | `slippage_fee_bps` | [1.0, 15.0] | Trading |
| 32 | `trading_max_position_pct` | [0.02, 0.20] | Trading |
| 33 | `trading_max_drawdown_pct` | [0.10, 0.35] | Trading |
| 34 | `buyback_pct` | [0.05, 0.50] | Buyback |
| 35 | `buyback_min_profit` | [100, 5000] | Buyback |
| 36 | `emission_sigmoid_midpoint` | [0.2, 0.8] | Emissions |
| 37 | `emission_sigmoid_steepness` | [1.0, 20.0] | Emissions |
| 38 | `l1_l2_emission_split` | [0.40, 0.80] | Emissions |
| 39 | `rate_limit_epoch_seconds` | [3600, 172800] | Rate limit |
| 40 | `feedback_min_l2_epochs` | [1, 10] | Feedback thresh |
| 41 | `feedback_bonus_threshold` | [0.4, 0.8] | Feedback thresh |

### Key Bittensor On-Chain Parameters (6 Primary)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `tempo` | 360 | [100, 1440] | Blocks between weight-setting |
| `immunity_period` | 5000 | [1000, 20000] | New neuron protection |
| `bonds_moving_avg` | 900000 | [500000, 999999] | Bond decay rate |
| `adjustment_alpha` | 0 | [0, 65535] | Weight adjustment speed |
| `max_weight_limit` | 65535 | [16384, 65535] | Max per-uid weight |
| `min_allowed_weights` | 1 | [1, 16] | Min weights per validator |

---

## Appendix B: Attack-Parameter Mapping

When a specific attack is breached, these parameters are most likely to
help defend against it:

| Attack | Primary Parameters | Secondary Parameters |
|--------|-------------------|---------------------|
| Overfitting | `overfit_gap_threshold`, `overfit_decay_rate`, `l1_overfitting_penalty` | `promotion_max_overfitting_score` |
| Plagiarism | `fingerprint_correlation_threshold` | `l1_feature_efficiency` |
| Single-metric gaming | All L1 weights (rebalance) | `l1_variance_score` |
| Sybil | `fingerprint_correlation_threshold` | On-chain `min_burn` |
| Copy-trading | `copy_trade_*` (all 3) | `l2_execution_quality` |
| Random discrimination | `l1_overfitting_penalty`, `l1_variance_score` | `emission_sigmoid_steepness` |
| Adversarial dominance | All L1 weights, `feedback_*` weights | `promotion_top_n` |
| Insufficient separation | `l1_overfitting_penalty`, feedback weights | `emission_sigmoid_steepness` |
| Score concentration | `emission_sigmoid_midpoint`, `emission_sigmoid_steepness` | `max_weight_limit` (on-chain) |

---

## Appendix C: Quick Commands

```bash
# Run single simulation (fast sanity check)
cd subnet && python -m tuning.orchestrator --mode single

# Run attack analysis (5 trials)
cd subnet && python -m tuning.orchestrator --mode attack --trials 10

# Run full NSGA-II optimization
cd subnet && python -m tuning.orchestrator --mode optimize --generations 50 --population 30

# Run testnet emulator (local chain)
cd subnet && python -m testnet.run_emulator --mode evolve --network local --netuid 1

# Run autoresearch experiment loop
cd subnet && python -m tuning.autoresearch_loop --budget-minutes 5 --max-experiments 100

# Check attack detection on current best config
cd subnet && python -m tuning.attack_detector

# Start monitoring
cd subnet && docker compose -f testnet/docker-compose.testnet.yml up -d
```

---

## NEVER STOP

Once the swarm is started, it runs **indefinitely** until manually
stopped. The operator may be asleep, away from the computer, or
otherwise unavailable. Do not pause to ask for confirmation. Do not ask
"should I continue?" — the answer is always yes.

If you run out of ideas, think harder. Re-read the scoring code. Look at
parameter interactions. Try combining near-misses. Try radical changes.
If the optimizer is plateauing, increase mutation rates. If attacks keep
being breached, study the attack detector code and find gaps.

The loop runs until the human interrupts you, period.
