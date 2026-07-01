# Insignia Subnet Emulator — Specification

**Version**: 1.0.0
**Status**: Authoritative spec for the agent-orchestrated emulator
**Scope**: How a swarm of orchestration agents deploys the Insignia subnet onto a
local Bittensor testnet, populates it with a suite of agent miners, and tunes the
incentive **weights and on-chain hyperparameters** in a closed iterative loop
until the subnet reaches a **continuously-improving steady state** that is
**provably resistant to incentive-mechanism gaming**.

This document is the emulator-level companion to:

- [SUBNET_SPEC.md](SUBNET_SPEC.md) — subnet identity and interfaces
- [PAIRING_MECHANISM.md](PAIRING_MECHANISM.md) — the paired genetic incentive mechanism (authoritative)
- [INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) — scoring vectors, attack analysis, commit-reveal
- [PARAMETER_TUNING_PLAN.md](PARAMETER_TUNING_PLAN.md) — the 94+ parameter optimization surface
- [TESTNET_DEPLOYMENT.md](TESTNET_DEPLOYMENT.md) — deployment runbook and defaults
- [../program.md](../program.md) — the agent swarm protocol (roster, reset protocol, hard rules)

External references (Bittensor):
[SDK API](https://docs.learnbittensor.org/sdk/bt-api-ref) ·
[btcli](https://docs.learnbittensor.org/btcli) ·
[extrinsics](https://docs.learnbittensor.org/subtensor-api/extrinsics) ·
[subtensor](https://github.com/opentensor/subtensor/)

---

## 1. Objective

> Find a configuration of the Insignia incentive mechanism — its scoring weights,
> defense thresholds, pairing knobs, and on-chain subnet hyperparameters — under
> which **honest contribution is the only profitable strategy**, and prove it by
> running the configuration against a continuously adversarial population on a
> real consensus chain until the system stops improving only because it has
> nothing left to fix.

The emulator is the falsification engine for that claim. Each iteration deploys a
candidate configuration to a local subtensor chain, lets honest *and* adversarial
agent miners compete under real Yuma consensus, measures whether any adversary
captured emission it did not earn, and feeds the result back into a
multi-objective optimizer. The loop halts only when a formal **convergence
contract** (§7) is unanimously satisfied — i.e. the subnet is in a
self-sustaining, attack-resistant steady state.

### Success is defined by four simultaneous conditions

| # | Condition | Operational meaning |
|---|-----------|---------------------|
| 1 | **Separation** | Honest miners out-earn every adversarial archetype by a stable margin. |
| 2 | **Breach-free** | Across all 19 surveillance vectors, no adversary earns emission disproportionate to value. |
| 3 | **Stability** | Honest score variance is low; no metric is fragile to seed/regime. |
| 4 | **Steady state** | The optimizer can no longer improve any objective without regressing another (Pareto-stationary). |

A configuration that maximizes honest performance but leaks emission to *any*
gaming strategy is a **failed** configuration, regardless of headline numbers.

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER (agent swarm)                     │
│   deployer · simulator · sentinel · tuner · researcher · coder            │
│   coordinate exclusively through the insignia-local MCP (swarm state)      │
└───────────┬───────────────────────────────┬──────────────────────────────┘
            │ reads/writes shared state       │ proposes configs / reads fitness
            ▼                                 ▼
┌───────────────────────────┐     ┌───────────────────────────────────────┐
│  insignia-local MCP        │     │            EMULATOR CORE               │
│  (MongoDB `swarm` DB)      │     │  testnet/emulator.py :: InsigniaEmulator│
│                            │     │                                        │
│  simulation_runs           │     │  decode(param_vector) → config         │
│  tuner_optimization        │◄───►│  spawn agent miners (honest+adversary) │
│  experiment_results        │     │  SimulationHarness.run()               │
│  sentinel_state            │     │  AttackDetector.evaluate()             │
│  convergence_criteria      │     │  compute_fitness()                     │
│  breach_trends             │     │  ChainInterface.set_weights() ─────────┼──► subtensor
│  chain_weights             │     │  export Prometheus metrics             │     (local
│  defense_strategies        │     │                                        │      chain)
│  approvals (HITL gate)     │     └───────────────────────────────────────┘
└───────────────────────────┘                    │
                                                  ▼
                                   ┌───────────────────────────┐
                                   │  LOCAL TESTNET (subtensor) │
                                   │  ws://<chain-host>:9945       │
                                   │  (pre-provisioned Docker)   │
                                   │  1 owner · 1 validator ·    │
                                   │  12 miner wallets · 1 netuid│
                                   └───────────────────────────┘
```

> **Environment addressing (read before deploying).** The orchestration plane,
> the chain, and the MCP store run on **separate hosts** — do **not** assume
> `127.0.0.1`. The deployer must address them explicitly:
>
> | Plane | Address | How the stack reaches it |
> |-------|---------|--------------------------|
> | Subtensor chain (Docker) | `ws://<chain-host>:9945` | `SUBTENSOR_LOCAL_ENDPOINT` env var → `config.endpoint` |
> | insignia-local MCP (MongoDB) | _(internal host — set via `MONGO_URI`, not published)_ | `MONGO_URI` env var (insignia-local MCP server) |
>
> The chain container is **already running** on `<chain-host>`; the deployer's job
> is to *connect to and verify* it, **not** to `docker compose up` a new chain on
> its own host (that was the cause of the cancelled/failed deploy tasks). If the
> container runs the upstream `localnet.sh` instead of this repo's
> `docker-compose.testnet.yml`, the WS port is `9946` — confirm with the
> connectivity check below before funding wallets.

Three planes, cleanly separated:

1. **Orchestration plane** — the agent swarm. Agents do not share memory in
   process; they share *durable state* through the insignia-local MCP. This makes
   the loop resumable, auditable, and safe to run unattended (per
   [program.md](../program.md) §13: *continue until interrupted*).
2. **Emulator plane** — deterministic Python that turns a parameter vector into a
   measured fitness vector against a real chain. This is the unit of evaluation.
3. **Chain plane** — a local subtensor node running real Yuma consensus, so the
   emulator measures *emission* (the thing adversaries actually want), not a
   proxy.

### 2.1 Why a real local chain, not pure simulation

Pure simulation can be gamed by the simulator's own simplifications. Running the
candidate config through `subtensor.set_weights(...)` and reading emission back
off the metagraph closes that gap: latency, weight-clipping
(`max_weight_limit`), bonds (`bonds_moving_avg`), `tempo`, and commit-reveal
windows behave as they will on mainnet. The emulator degrades gracefully to
`offline` mode when the SDK/chain is unavailable (see
[emulator.py](../testnet/emulator.py) `ChainInterface.connect`), but the
acceptance gates in §9 must be cleared in `online` mode.

---

## 3. The orchestration loop

One **iteration** of the master loop is:

```
                ┌────────────────────────────────────────────────┐
                │ 1. tuner: pull Pareto state from MCP            │
                │    propose population of candidate configs      │
                │    (scoring weights + defense + pairing +       │
                │     on-chain hyperparameters)                   │
                └───────────────────────┬────────────────────────┘
                                        ▼
                ┌────────────────────────────────────────────────┐
                │ 2. deployer: ensure chain up, wallets funded,   │
                │    subnet registered, hyperparameters applied   │
                │    via btcli (HITL-gated, see §6.4)             │
                └───────────────────────┬────────────────────────┘
                                        ▼
                ┌────────────────────────────────────────────────┐
                │ 3. simulator: for each candidate, run emulator  │
                │    epoch with full agent-miner population       │
                │    (honest + adversarial). set_weights on-chain.│
                └───────────────────────┬────────────────────────┘
                                        ▼
                ┌────────────────────────────────────────────────┐
                │ 4. sentinel: evaluate 19 attack vectors,        │
                │    breach_rate, separation, integrity score;    │
                │    classify INFO/WARNING/CRITICAL/EMERGENCY;     │
                │    fire reset triggers if needed                │
                └───────────────────────┬────────────────────────┘
                                        ▼
                ┌────────────────────────────────────────────────┐
                │ 5. tuner: fold fitness back into NSGA-II;        │
                │    update Pareto front + knee point in MCP      │
                └───────────────────────┬────────────────────────┘
                                        ▼
                ┌────────────────────────────────────────────────┐
                │ 6. researcher: if stagnating, escalate radical  │
                │    level; propose new detectors/scoring ideas;  │
                │    log experiment rows to MCP + experiments.tsv │
                └───────────────────────┬────────────────────────┘
                                        ▼
                ┌────────────────────────────────────────────────┐
                │ 7. evaluate convergence contract (§7).          │
                │    met → steady state reached → hold + validate │
                │    not met → loop                               │
                └────────────────────────────────────────────────┘
```

The loop is **continuous and self-pacing**. It does not stop on a fixed
generation count; it stops on the convergence contract, and even then transitions
into an empirical-validation hold (re-running the converged config under fresh
seeds to confirm the steady state is real, not a surrogate artifact).

### 3.1 Agent roster (orchestration plane)

Reuses the six roles defined in [program.md](../program.md) §3. Emulator-specific
responsibilities:

| Agent | Owns in the emulator loop | Primary code / MCP surface |
|-------|---------------------------|----------------------------|
| **deployer** | Chain lifecycle, wallet funding, subnet registration, applying on-chain hyperparameters via btcli, Prometheus/Grafana stack | `testnet/subnet_manager.py`, `testnet/wallet_manager.py`, `testnet/scripts/*.sh`; MCP `deployer_state`, `chain_weights` |
| **simulator** | Instantiating the agent-miner population, running `SimulationHarness`/`InsigniaEmulator` epochs, emitting telemetry | `tuning/simulation.py`, `testnet/emulator.py`; MCP `simulation_runs`, `simulation_epochs`, `simulation_results` |
| **sentinel** | Attack-vector evaluation, breach-trend tracking, alert classification, reset recommendations, convergence gating | `tuning/attack_detector.py`, `tuning/composite_integrity_scorer.py`; MCP `sentinel_state`, `breach_trends`, `breach_alerts`, `convergence_metrics` |
| **tuner** | NSGA-II over mechanism parameters, Pareto-front and knee-point maintenance, surrogate modeling | `tuning/optimizer.py`, `tuning/parameter_space.py`; MCP `tuner_optimization`, `optimization_runs` |
| **researcher** | Autoresearch experiments, radical-level escalation, new defense ideas, keep/discard accounting | `tuning/autoresearch_loop.py`; MCP `experiment_results`, `researcher_insights`, `researcher_state` |
| **coder** | Implementing missing detectors/telemetry/scripts the researcher proposes | repository-wide; MCP `tasks`, `defense_strategies` |

### 3.2 insignia-local MCP as the orchestration substrate

The MCP server exposes the MongoDB `swarm` database plus a human-in-the-loop
approval API. It is the **single source of truth for run state** across the
agents and across restarts. The emulator and agents must:

- **Persist** every epoch/experiment/optimizer step so the loop is resumable.
- **Read** prior Pareto state, seed lineage, and sentinel posture before proposing
  the next move (no re-deriving from scratch).
- **Gate** chain-mutating and irreversible actions behind the approval API.

Key collections (observed live; treat as the canonical run ledger):

| Collection | Role in the loop |
|------------|------------------|
| `convergence_criteria` | Formal definition of "steady state" (the convergence contract, §7). |
| `simulation_runs`, `simulation_epochs`, `simulation_results`, `simulation_agents` | Per-run / per-epoch / per-agent telemetry from the emulator. |
| `tuner_optimization`, `optimization_runs` | NSGA-II population, Pareto front, knee points, hypervolume history. |
| `experiment_results`, `experiment_configs`, `experiment_tsv_tracking` | Researcher loop ledger (mirrors `results/experiments.tsv`). |
| `sentinel_state`, `breach_trends`, `breach_alerts`, `attack_monitoring`, `simulation_vector_severities` | Attack surveillance posture and per-vector severity over time. |
| `composite_integrity_scores` | EXP-023 integrity scoring (§7 stability signal). |
| `chain_weights` | Weights actually set on-chain per epoch — the emission ground truth. |
| `defense_strategies` | Catalog of deployed/candidate defenses (e.g. PC-VH-006). |
| `convergence_metrics`, `convergence_state`, `reset_checkpoints`, `state_preservation` | Convergence state machine + reset/rollback points. |
| `agent_memory`, `audit_log` | Cross-agent memory and a full audit trail of every action. |
| `approvals` | Pending/decided human-in-the-loop approvals (the HITL gate). |

> **Hard rule (inherited from program.md §13):** the MCP ledger and the repo
> must not drift. When the emulator computes a result it persists it; when an
> agent acts on state it cites the document it read. `audit_log` is append-only.

---

## 4. Local testnet deployment

**Scope: local testnet only.** For now the emulator targets the
**local subtensor** (`NetworkTarget.LOCAL`) exclusively. Do **not** deploy to the
public Bittensor testnet (`test.finney`, `NetworkTarget.TESTNET`) — that path is
out of scope until the local loop is validated.

The chain is a **pre-provisioned Docker subtensor on `<chain-host>`**, not a chain
the deployer brings up locally. The whole stack — `subnet_manager`,
`wallet_manager`, and the emulator's `ChainInterface` — derives its endpoint from
`config.endpoint`, which reads `SUBTENSOR_LOCAL_ENDPOINT`
(see [testnet/config.py](../testnet/config.py), default `ws://127.0.0.1:9945`).
So the **only** thing needed to point everything at the real chain is to export
that variable before running anything:

```bash
export SUBTENSOR_LOCAL_ENDPOINT="ws://<chain-host>:9945"   # 9946 if upstream localnet.sh
```

No code change is required; the previous deploy failures were purely the default
`127.0.0.1` endpoint resolving to nothing on the deployer's host.

### 4.1 Topology

| Component | Count | Wallet convention |
|-----------|------:|-------------------|
| Owner | 1 | `insignia-owner` / `default` |
| Validator | 1 | `insignia-validator` / `default` |
| Miner | 12 | `insignia-miner-{i}` / `default` |

The 12 miner wallets host the agent-miner suite (§5). One UID space, one
metagraph, one `set_weights` vector — per the paired genetic mechanism
([PAIRING_MECHANISM.md](PAIRING_MECHANISM.md) §5).

### 4.2 Deploy sequence (deployer agent)

The chain already exists on `<chain-host>`, so the sequence **connects and verifies**
rather than provisioning a chain. Pass the endpoint explicitly — the check scripts
take it as their first argument.

```bash
cd subnet
export SUBTENSOR_LOCAL_ENDPOINT="ws://<chain-host>:9945"   # 9946 if upstream localnet.sh

# 1. Verify connectivity to the EXISTING Docker chain (do NOT docker compose up here)
bash testnet/scripts/check_chain_connectivity.sh "$SUBTENSOR_LOCAL_ENDPOINT"
bash testnet/scripts/check_wallet_balances.sh   "$SUBTENSOR_LOCAL_ENDPOINT"
#   If connectivity fails: the container is unreachable or on a different port —
#   resolve addressing with the infra owner; do NOT fall back to a local chain.

# 2. Create wallets, register subnet, fund miners/validator (HITL-gated, §6.5)
python -m testnet.wallet_manager          # coldkeys/hotkeys per convention
python -m testnet.subnet_manager create   # btcli subnet create on the local chain
python -m testnet.subnet_manager register # register validator + 12 miners

# 3. Run the emulator loop against the live netuid
python -m tuning.orchestrator --mode testnet --network local \
       --netuid <NETUID> --generations 20 --population 30
```

> Monitoring (Prometheus/Grafana) may be brought up separately via
> `docker compose -f testnet/docker-compose.testnet.yml up -d` **only if** that
> stack is not already running on the chain host; it is not required to reach the
> chain itself.

The emulator's `ChainInterface.set_weights` normalizes the credit vector and
calls `subtensor.set_weights(netuid, uids, weights, wait_for_inclusion=True)`
once per epoch (Yuma consensus). See the
[extrinsics reference](https://docs.learnbittensor.org/subtensor-api/extrinsics)
for `set_weights` / commit-reveal primitives.

### 4.3 On-chain hyperparameters (tuned, not just defaulted)

These are first-class members of the optimization surface, applied via btcli by
the deployer and recorded in MCP `deployer_state` before each evaluation block.
Defaults from `SubnetHyperparameters` ([config.py](../testnet/config.py)):

| Hyperparameter | Default | Why it matters to anti-gaming |
|----------------|--------:|-------------------------------|
| `tempo` | 360 | Epoch length; too short amplifies latency-arbitrage surface. |
| `immunity_period` | 5000 | Protects new honest miners from premature deregistration. |
| `max_weight_limit` | 65535 | Caps any single UID's emission share (anti-concentration). |
| `min_allowed_weights` | 1 | Forces validators to spread weight (anti-collusion). |
| `adjustment_alpha` | 0 | Registration difficulty smoothing (raises Sybil cost). |
| `bonds_moving_avg` | 900000 | Smooths validator bonding; dampens collusive weight spikes. |
| `commit_reveal_weights_enabled` | true | On-chain commit-reveal for weights (latency-arbitrage defense). |
| `commit_reveal_weights_interval` | 1000 | Reveal cadence for committed weights. |
| `alpha_high` / `alpha_low` | 58982 / 45875 | Liquid-alpha bonds bounds. |
| `liquid_alpha_enabled` | true | Bond responsiveness. |

The 6 most operationally relevant for subnet-owner tuning are `tempo`,
`immunity_period`, `min_allowed_weights`, `max_weight_limit`, `adjustment_alpha`,
`bonds_moving_avg` (see [PARAMETER_TUNING_PLAN.md](PARAMETER_TUNING_PLAN.md)).
The deployer reads target values from the tuner's proposed config and applies
them with `btcli subnets hyperparameters` (see
[btcli docs](https://docs.learnbittensor.org/btcli)).

---

## 5. The agent-miner suite

The emulator replaces real network participants with parameterized bots so the
optimizer can dial attack intensity. Each archetype exists to *try to break* a
specific defense; a configuration is only accepted if every adversary fails to
out-earn honest miners.

### 5.1 Researcher-side archetypes (model submitters)

| Archetype | Strategy it probes | Defense under test |
|-----------|--------------------|--------------------|
| **Honest** | Best-effort generalizing model | Baseline / separation signal |
| **Overfitter** | Memorizes public data correlated to the window | Overfitting penalty + proprietary holdout + rolling windows |
| **Copycat** | Re-serializes/perturbs another miner's model | `ModelFingerprinter` + prediction correlation + code fingerprint |
| **SingleMetricGamer** | Maxes one metric (e.g. F1), ignores rest | 7-metric composite (max weight 22%) |
| **Sybil** | N correlated identities | Fingerprint + correlation + staking cost + PC-VH-006 symbol diversity |
| **Random** | Noise submissions | Noise-floor baseline |

### 5.2 Trader-side archetypes (strategy operators)

| Archetype | Strategy it probes | Defense under test |
|-----------|--------------------|--------------------|
| **HonestTrader** | Best-effort strategy on assigned model | Baseline / separation |
| **CopyTrader** | Mirrors another trader's positions | `CopyTradeDetector` (time/size/correlation) |
| **(harness) Manipulator / Arbitrage / FrontRunner** | Wash trading, spoofing, latency arbitrage, post-hoc submission | Execution-quality scoring, commit-reveal, latency penalty, `latency_arbitrage_pairing` vector |

### 5.3 Default operational mix (the harder operating environment)

Per [program.md](../program.md) §4 / [config.py](../testnet/config.py):

- 6 honest + 4 adversarial researchers
- 3 honest + 1 adversarial traders
- **14-agent benchmark population**, mapped onto the 12 miner wallets + role pairing

### 5.4 Stable per-run model routing (intelligence diversity)

Each agent is assigned one stable external MCP model route for the whole run
(`ModelRoutingConfig`, seeded, reproducible, emitted in telemetry). This models
decentralized intelligence diversity **without** injecting per-epoch routing
noise that would contaminate fitness. Route assignment is orthogonal to
archetype: an adversary does not become honest by drawing a better route.

---

## 6. Optimization & anti-gaming proof

### 6.1 What gets tuned

Two coupled levels (full enumeration in
[PARAMETER_TUNING_PLAN.md](PARAMETER_TUNING_PLAN.md)):

- **Application-level (emulator-tuned):** 7 model weights, 9 trading weights,
  overfitting detector, pairing knobs (`partners_per_miner`, `elite_fraction`,
  `mutation_rate`, `pair_blend_alpha`, `marginal_contribution_weight`,
  `fixed_pair_correlation_threshold`, `pairing_seed_source`, `max_pairs`),
  commit-reveal timing, validation-timing penalties, consensus-integrity
  thresholds, symbol-diversity (PC-VH-006), economic-mechanism knobs, model
  routing.
- **On-chain (btcli-applied):** the subnet hyperparameters in §4.3.

Constraints enforced by the repair operator: model weights sum to 1.0, trading
weights sum to 1.0, all weights ∈ [0.01, 0.50], thresholds positive.

### 6.2 Objectives (NSGA-II, all minimized)

1. `-mean_honest_score` — maximize honest performance.
2. `attack_breach_rate` — minimize fraction of the 19 vectors breached.
3. `score_variance` — minimize honest-score instability.
4. `-score_separation` — maximize honest-vs-adversarial gap.

Multi-objective by design: a high-return config with deep drawdowns or any breach
does **not** dominate a steadier, breach-free one, so the Pareto front retains a
diversity of viable mechanisms rather than collapsing onto a single gameable
metric.

### 6.3 The anti-gaming threat model (the core of the spec)

The subnet must satisfy one invariant:

> **Emission ∝ value.** No miner can increase its emission share without
> increasing the genuine, transferable value it contributes. Every known shortcut
> to emission is closed structurally (by the mechanism), not just penalized after
> the fact.

The three classes the user named — **overfit**, **latency arbitrage**,
**collusion** — plus their relatives, each map to a structural defense whose
effectiveness the emulator continuously re-measures:

#### A. Overfitting / benchmark gaming

| Shortcut | Why it fails (structural) | Emulator check |
|----------|---------------------------|----------------|
| Memorize public data that correlates with the eval window | Validators score against **proprietary tick data** miners can't access; rolling holdout windows rotate per epoch; overfitting penalty targets IS/OOS gap; Variance Score punishes regime-specific fit | Overfitter archetype must rank **below** honest mean; vector `overfitting_exploitation` not breached |
| Win one lucky window | Every metric is **variance-penalized** (`mean − λ·std`) across rolling sub-windows | `score_variance` objective; consistency metric (20% weight) |

#### B. Latency arbitrage / timing manipulation

| Shortcut | Why it fails (structural) | Emulator check |
|----------|---------------------------|----------------|
| Submit predictions after market data materializes, during a slow validator window | **Commit-reveal** binds the hash before data exists (commit closes T-5s, reveal T+5s..T+20s); on-chain `commit_reveal_weights_enabled` binds weights too | `validator_latency_exploitation` & `prediction_timing_manipulation` severity < target; `commit_reveal_effectiveness ≥ 0.667` floor |
| Time submissions to a known partner/validator | Partner identity is **chain-seeded and revealed only at evaluation time**; `min_prediction_lead_time` rejects too-close submissions; high-latency validator scores are discounted | `latency_arbitrage_pairing` vector not breached |

#### C. Collusion (miner↔miner and miner↔validator)

| Shortcut | Why it fails (structural) | Emulator check |
|----------|---------------------------|----------------|
| Researcher + trader agree to only look good together | **No self-selection** — pairing is chain-seeded; **K-partner floor** forces evaluation against unchosen partners; **variance-penalized marginal-contribution credit** makes non-transferable lift unprofitable; `CollusionGraphDetector` flags the interaction anomaly directly | Colluding-pair archetypes earn **below** honest; `pair_collusion` / `partner_selection_gaming` not breached |
| Sybil cluster splits one entity across many UIDs | Fingerprint + prediction-correlation catch behavioral clones; staking cost scales linearly while per-identity reward falls; PC-VH-006 caps symbol dominance | `sybil_attack` severity declining; Sybil archetype share < 2× single-honest share |
| Validator inflates a friend's weight | Validators **no longer choose pairings/promotion** (lever removed); weight-entropy minimum, cross-validator variance cap, rotation limits, agreement threshold, temporal-correlation monitoring; deterministic pairing means honest validators reproduce identical scores, so a deviating validator is a detectable cross-validator anomaly | `miner_validator_collusion`, `weight_entropy_violation`, `cross_validator_score_variance`, `validator_agreement_anomaly` not breached |

#### D. Single-metric gaming, plagiarism, spam, paper-trade fabrication

Covered by composite scoring (no metric > 22%/20%), SHA-256 + behavioral
fingerprinting with **shared rewards for correlated work** (copying is strictly
dominated), 1-submission-per-epoch rate limiting, and continuous independent
position verification. Full matrix in
[INCENTIVE_MECHANISM.md](INCENTIVE_MECHANISM.md) §"Attack Vector Analysis".

### 6.4 Why these are *proofs*, not assertions

Each row above is a **falsifiable hypothesis** the emulator re-tests every
iteration: the corresponding adversarial archetype is *in the population*, and the
sentinel computes a breach boolean + severity for its vector. A defense is
"proven" for a given config only while:

1. its adversary archetype scores strictly below the honest mean (separation), **and**
2. its vector's breach severity stays under threshold across reruns (robustness), **and**
3. the result holds **on-chain** (real emission), not just in offline sim.

If any of these fail, the sentinel raises an alert and the config is rejected.
This makes "ungameable" an empirically maintained property, not a one-time claim —
exactly the standing the swarm must defend (program.md §6).

### 6.5 Human-in-the-loop approval gate

Irreversible or outward-facing actions are **not** auto-executed. The deployer
posts an approval request to the MCP (`list_pending_approvals` /
`approve_action` / `reject_action`) and waits. Gated actions include:

- Applying on-chain hyperparameter changes via btcli.
- Subnet creation / deregistration.
- Any FULL reset (tear-down + rebuild, program.md §9).
- Promoting a converged config as the production reference.

Everything inside the offline/online evaluation loop (proposing configs, running
sims, setting testnet weights) runs autonomously.

### 6.6 Surrogate vs. empirical adversary model (the validation contract)

> **Lesson from the V13-R3 invalidation (Orchestration Report 2026-06-29T01-35-48).**
> NSGA-II v13 R3 converged to a knee that *predicted* `0.963` honest/adversarial
> separation and was promoted as state-of-record. When validated against the full
> adversarial population, empirical separation was `~0.23`: the best adversary
> (Copycat) scored `0.733` — not the `~0.018` the prediction implied (a ~40×
> underestimate) — and adversaries captured ~64.7% of chain weight. The config
> **failed** the `≥0.90` separation gate.

> **Correction (2026-07-01, code audit).** The original §6.6 attributed the
> divergence to a "GP surrogate trained on a simplified analytical adversary
> model." A code audit of `subnet/tuning/optimizer.py:172-213`
> (`InsigniaTuningProblem._evaluate`) found **no surrogate exists in the codebase**
> — the optimizer calls `SimulationHarness.run()` directly per candidate, and
> `compute_fitness` (`optimizer.py:88-112`) reads `sim_result.adversarial_researcher_scores`
> straight from the harness. The "GP surrogate," "analytical adversary model,"
> "R²=0.96," and "ExpectedImprovement infill" described in earlier docs and in
> `reference_configs/knee_point_V13-R3.json` are metadata only — none of that
> machinery is implemented. The §6.6 contract clause "the optimizer's adversary
> scores must come from the same evaluation path as the acceptance gates" was
> therefore **already satisfied by construction**. The real root cause is below.

The failure was **not** in the metric definition (`separation = honest − adv`,
which is consistent) and **not** in a surrogate. It was in the *harness's
incomplete anti-gaming penalty coverage*:

1. **Incomplete penalty coverage (primary).** `subnet/tuning/simulation.py:870-873`
   applies an anti-copy multiplier only to `CopycatMiner` and `CopyTrader`. The
   other adversaries (`SybilMiner`, `OverfittingMiner`, `SingleMetricGamer`,
   `PartnerGamer`) have **no penalty path** in the scoring loop and score
   ~0.90 — the same as honest. `SybilMiner` actually scores *higher* than honest
   (0.9163 vs 0.9151) because the harness computes `sybil_pressure` and
   `ensemble_signals` but never feeds them back into `miner_scores`. This is the
   real separation leak.
2. **Empirical reproduction.** `tests/test_simulation_separation.py` reproduces
   the V13-R3 invalidation numbers against the current harness: with the prior
   `0.50` multiplier, separation = 0.1510 (honest 0.9007, adversarial 0.7498).
   Tightening the multiplier to `0.10` only raised separation to 0.2195 —
   Copycat/CopyTrader dropped, but the other adversaries were unchanged. The
   `0.733` "best adversary" figure from the invalidation report is consistent
   with the harness's adversarial mean and was real; it was just attributed to
   the wrong cause.
3. **Pareto collapse (symptom, not cause).** All 26 Pareto solutions sat in a
   narrow `0.958–0.967` separation band — a tell-tale of a false local optimum.
   With no surrogate in the loop, the collapse is explained by the optimizer
   maximizing separation against a harness in which 4 of 6 adversary types
   score ~0.90 regardless of parameters: there is no parameter direction that
   reduces their scores, so the optimizer converged on the only lever it had
   (Copycat/CopyTrader, via the multiplier).

**Contract the loop must honor:**

- **The optimizer's adversary scores already come from the harness** — this
  clause is satisfied by `optimizer.py:190-200`. No change needed.
- **The harness must penalize every adversary type enumerated in §6.3, not
  just Copycat/CopyTrader.** Each adversary archetype in §5.1 / §5.2 must have
  a corresponding penalty path in `simulation.py`'s scoring loop. The
  `test_no_adversary_outscores_honest_mean` test in
  `tests/test_simulation_separation.py` pins this: no adversary type may score
  higher than the honest mean. Currently `sybil` leaks.
- **No surrogate-only promotion.** A predicted gate pass (§9) is provisional
  until the gates clear on empirical (harness, then online) scores across ≥2
  seeds. Record both the predicted and empirical vectors so divergence is
  auditable. (This clause is now vacuous for the harness-direct optimizer —
  there is no surrogate to diverge from — but is retained for any future
  surrogate-assisted search.)
- **Divergence is a sentinel signal.** If predicted-vs-empirical separation
  (or any gate metric) diverges beyond a small tolerance, treat the run as
  `diverged` (§7.1) and do not promote until the gap closes.
- **Harness code stability is a convergence precondition.** The convergence
  contract (§7) must not be declared met while `compute_fitness` is fed by a
  harness with known unpunished adversary types. A `diverged` sentinel must
  fire if `test_no_adversary_outscores_honest_mean` fails.

---

## 7. The convergence contract (continuously-improving steady state)

"Continuously improving until steady state" is given a **formal, unanimous**
definition stored in MCP `convergence_criteria` (`insignia_subnet_convergence_v1`).
The loop stops only when **all** enabled criteria are met for the configured
grace period (`require_unanimous: true`, `grace_period: 5`):

| Criterion | Signal | Met when |
|-----------|--------|----------|
| **Pareto-front stability** | `hypervolume_delta` | No significant improvement (< 0.01) for 25 consecutive generations |
| **Attack-vector convergence** | `attack_pattern_entropy` variance | Variance < 0.05 for 15 consecutive evaluations |
| **Fitness plateau** | `mean_population_fitness` | Improvement < 0.005 for 20 consecutive generations |
| **Parameter-space saturation** | coverage | > 85% coverage with > 70% unique parameter combinations |

A second contract governs the **researcher** auto-cycler
(`gbdt_autoresearch_convergence`): a 6-state machine
(`not_started → monitoring → stagnating → escalated → converged|diverged`) with
stagnation detection (3 cycles, hypervolume window), early stopping (patience 15,
max 1000 generations), and **escalation on stagnation** (radical levels 1–5,
de-escalation on improvement). This is what makes the loop "continuously
improving": when progress stalls, the researcher *escalates* (boundary expansion
→ scoring modification → detection heuristic → architecture redesign) rather than
declaring premature victory.

> **Steady state ≠ frozen.** Convergence means the optimizer cannot improve any
> objective without regressing another *and* the adversarial entropy has settled —
> i.e. there are no remaining unexploited gaming strategies and no free
> performance left. The system then enters an empirical-validation hold (re-run
> the converged config under fresh seeds; confirm breach_rate and
> commit-reveal effectiveness hold) before it is eligible for the production-
> reference approval gate.

### 7.1 Divergence and reset protocol

If sentinel detects breach severity rising or hypervolume decreasing for 10
cycles → `diverged`. Reset workflows (program.md §9):

- **SOFT** — double breach-rate emphasis, inject 5 random candidates, keep elites.
- **HARD** — keep 30% Pareto elites + 30% researcher-best + 40% random; reset
  convergence counters; restart generations.
- **FULL** — save state to `state_preservation`, tear down & rebuild local infra,
  restart from a fresh population (HITL-gated).

---

## 8. Observability

Prometheus on `:8001/metrics` (emulator) / `:8000` (offline tuner); Grafana on
`:3000`; Prometheus server on `:9090`. Core series to preserve
([program.md](../program.md) §8):

- Scoring: `insignia_researcher_composite_score`, `insignia_trader_composite_score`
- Attacks: `insignia_attack_breach`, `insignia_attack_severity`, `insignia_total_breaches`
- Optimizer: `insignia_best_fitness`, `insignia_pareto_front_size`
- Commit-reveal / timing: `insignia_commit_timestamp`, `insignia_reveal_timestamp`,
  `insignia_no_reveal_streak`, `insignia_timing_attack_composite_severity`,
  `insignia_commit_reveal_effectiveness`
- Diversity: `insignia_trading_pair_activity`, `insignia_btc_eth_dominance_ratio`,
  `insignia_symbol_diversity_enforcement`, `insignia_ensemble_signal`

Every metric also lands in the MCP ledger so the loop is reconstructable from
state alone (no reliance on a live Prometheus instance for resume).

---

## 9. Acceptance gates (definition of done)

A configuration is promotable to the production-reference approval gate **only
when all** hold, in `online` mode, across ≥ 2 reruns with different seeds:

| Gate | Threshold |
|------|----------:|
| Honest mean score | ≥ 0.97 |
| Attack breach_rate | ≤ 5e-6 (target; floor: 0 breached vectors at WARNING+) |
| Honest/adversarial separation | ≥ 0.90 |
| Honest score variance | ≤ 0.002 |
| Commit-reveal effectiveness | ≥ 0.667 floor (≥ 0.76 target) |
| Validator-latency severity | < 0.05 |
| Prediction-timing severity | < 0.03 |
| Consecutive clean validations | ≥ 6 |
| Convergence contract (§7) | unanimously met + grace period |
| Sentinel posture | `SECURE_AND_IMPROVING` or stronger |

The bar is "match or beat the proven state of record." The last
**not-contradicted** checkpoint is the R2 knee `V13-R2-KP-020-a7f2` (honest
`0.9795`, breach `3.5e-6`, separation `0.953`). ⚠️ The newer
`V13-R3-KP-020-a3c7` knee **looked** better on paper (honest `0.9808`, breach
`2.6e-6`, separation `0.963`) but **failed empirical validation on separation**
(`~0.23` vs gate `≥0.90`) — see §6.6 and
[../reference_configs/knee_point_V13-R3.json](../reference_configs/knee_point_V13-R3.json).
**A surrogate-predicted gate pass is not a pass.** Promotion requires the gates
to clear on *empirical* (harness/online) scores, per §6.6 and §11.

---

## 10. Operational runbook

```bash
cd subnet

# Sanity: one offline simulation with defaults
python -m tuning.orchestrator --mode single

# Adversarial attack analysis across trials
python -m tuning.orchestrator --mode attack --trials 10

# Full offline evolutionary tuning
python -m tuning.orchestrator --mode optimize --generations 20 --population 30

# On-chain emulator loop (local testnet — pre-provisioned Docker chain on <chain-host>)
export SUBTENSOR_LOCAL_ENDPOINT="ws://<chain-host>:9945"   # 9946 if upstream localnet.sh
python -m tuning.orchestrator --mode testnet --network local --netuid <NETUID> \
       --generations 20 --population 30

# Continuous researcher loop (escalates on stagnation)
python -m tuning.autoresearch_loop --max-experiments 25

# Tests
python -m unittest discover -s tests -p "test_*.py"
```

Chain checks (against the existing remote chain — pass the endpoint explicitly):

```bash
export SUBTENSOR_LOCAL_ENDPOINT="ws://<chain-host>:9945"
bash testnet/scripts/check_chain_connectivity.sh "$SUBTENSOR_LOCAL_ENDPOINT"
bash testnet/scripts/check_wallet_balances.sh   "$SUBTENSOR_LOCAL_ENDPOINT"

# Monitoring stack (only if not already running on the chain host):
docker compose -f testnet/docker-compose.testnet.yml up -d
```

---

## 11. Hard rules (emulator)

1. A config that leaks emission to **any** adversary archetype is rejected, no
   matter how high its honest score — anti-gaming dominates performance.
2. Acceptance gates (§9) must be cleared in **online** (real-chain) mode, not just
   offline sim. Offline is the harder *stress* environment; chain is the *truth*.
3. Every evaluation, breach report, and optimizer step is persisted to the
   insignia-local MCP **before** acting on it; `audit_log` is append-only.
4. Chain-mutating and irreversible actions pass through the HITL approval gate.
5. Never remove a defense or risk control to make a number look better
   (program.md §13); deprecate only via a deliberate, recorded migration.
6. Keep this spec, [program.md](../program.md), and the code in sync; when they
   conflict, update docs to match code unless a migration is explicitly underway.
7. The loop continues until the convergence contract is met or it is interrupted;
   it does not pause for permission except at the HITL gate.
```
