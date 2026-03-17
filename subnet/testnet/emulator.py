"""
Testnet Emulator

The core emulator that bridges the Insignia simulation framework with a
real (or local) Bittensor subtensor chain. This enables:

  1. Running simulated L1/L2 miner agents against the actual Yuma consensus
  2. Setting weights on-chain and observing emission distribution
  3. Iterating over parameter configurations with real consensus feedback
  4. Evaluating attack resilience under realistic network conditions

Architecture:
                                ┌──────────────┐
    ┌──────────┐  submit models │              │  set weights
    │ L1 Agents├───────────────►│  Emulator    ├──────────────► Subtensor
    └──────────┘                │  Validator   │◄──────────────  (chain)
    ┌──────────┐  submit trades │              │  query metagraph
    │ L2 Agents├───────────────►│              │
    └──────────┘                └──────┬───────┘
                                       │
                               ┌───────▼───────┐
                               │ Parameter      │
                               │ Tuning Loop    │
                               │ (NSGA-II)      │
                               └───────────────┘

The emulator can run in two modes:
  - "offline": Full simulation without chain interaction (uses existing SimulationHarness)
  - "online":  Simulation with on-chain weight-setting via bittensor SDK
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insignia.scoring import CompositeScorer, WeightConfig, ReferenceOverfittingDetector
from insignia.incentive import (
    ModelFingerprinter,
    CopyTradeDetector,
    CrossLayerFeedbackEngine,
)
from insignia.cross_layer import CrossLayerOrchestrator, PromotionEngine, PromotionConfig

from neurons.l1_validator import L1Validator, ModelEvaluator, DemoBenchmarkProvider
from neurons.l2_validator import L2Validator

from tuning.parameter_space import (
    decode,
    encode_defaults,
    summarize_config,
    N_PARAMS,
    PARAM_NAMES,
)
from tuning.simulation import (
    SimulationHarness,
    SimulationResult,
    create_default_agents,
    MinerAgent,
    TraderAgent,
)
from tuning.attack_detector import AttackDetector, BreachReport
from tuning.optimizer import compute_fitness, OBJECTIVE_NAMES
from tuning.metrics_exporter import export_simulation_metrics

from .config import EmulatorConfig, NetworkTarget

logger = logging.getLogger("emulator")


@dataclass
class EmulatorEpochResult:
    """Results from a single emulator epoch."""

    epoch: int
    param_vector: Optional[np.ndarray] = None
    config_summary: str = ""

    sim_result: Optional[SimulationResult] = None
    breach_report: Optional[BreachReport] = None
    fitness: Optional[np.ndarray] = None

    on_chain_weights: Dict[str, float] = field(default_factory=dict)
    metagraph_snapshot: Optional[str] = None
    chain_block: int = 0

    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "epoch": self.epoch,
            "config_summary": self.config_summary,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "chain_block": self.chain_block,
        }

        if self.fitness is not None:
            result["fitness"] = {
                name: round(float(val), 6)
                for name, val in zip(OBJECTIVE_NAMES, self.fitness)
            }

        if self.sim_result:
            result["l1_scores"] = {
                uid: round(s, 6)
                for uid, s in self.sim_result.miner_scores.items()
            }
            result["l2_scores"] = {
                uid: round(s, 6)
                for uid, s in self.sim_result.l2_scores.items()
            }
            result["honest_l1_mean"] = (
                round(float(np.mean(self.sim_result.honest_l1_scores)), 6)
                if self.sim_result.honest_l1_scores
                else 0.0
            )
            result["adversarial_l1_mean"] = (
                round(float(np.mean(self.sim_result.adversarial_l1_scores)), 6)
                if self.sim_result.adversarial_l1_scores
                else 0.0
            )

        if self.breach_report:
            result["breach_report"] = self.breach_report.to_dict()

        if self.on_chain_weights:
            result["on_chain_weights"] = {
                k: round(v, 6) for k, v in self.on_chain_weights.items()
            }

        return result


@dataclass
class EmulatorRunResult:
    """Aggregate results from a full emulator run."""

    config: Dict[str, Any] = field(default_factory=dict)
    epochs: List[EmulatorEpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_fitness: Optional[np.ndarray] = None
    best_config: Optional[Dict[str, Any]] = None
    total_elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "config": self.config,
            "n_epochs": len(self.epochs),
            "best_epoch": self.best_epoch,
            "total_elapsed_seconds": round(self.total_elapsed, 1),
        }
        if self.best_fitness is not None:
            result["best_fitness"] = {
                name: round(float(val), 6)
                for name, val in zip(OBJECTIVE_NAMES, self.best_fitness)
            }
        result["epochs"] = [e.to_dict() for e in self.epochs]
        return result


class ChainInterface:
    """
    Abstraction over the Bittensor SDK for on-chain operations.

    In "online" mode, uses the bittensor Python SDK to set weights,
    query the metagraph, and interact with subtensor.
    In "offline" mode, all chain operations are no-ops.
    """

    def __init__(self, config: EmulatorConfig):
        self.config = config
        self._subtensor = None
        self._wallet = None
        self._online = False

    def connect(self) -> bool:
        """Attempt to connect to the subtensor chain."""
        try:
            import bittensor as bt

            if self.config.network == NetworkTarget.TESTNET:
                self._subtensor = bt.subtensor(network="test")
            else:
                self._subtensor = bt.subtensor(chain_endpoint=self.config.endpoint)

            self._wallet = bt.wallet(
                name=self.config.wallets.owner_coldkey,
                hotkey=self.config.wallets.owner_hotkey,
            )

            block = self._subtensor.block
            logger.info(
                "Connected to %s at block %d",
                self.config.endpoint,
                block,
            )
            self._online = True
            return True

        except ImportError:
            logger.warning(
                "bittensor SDK not available — running in offline mode. "
                "Install with: pip install bittensor"
            )
            return False
        except Exception as e:
            logger.warning(
                "Could not connect to %s: %s — running in offline mode",
                self.config.endpoint,
                e,
            )
            return False

    @property
    def is_online(self) -> bool:
        return self._online

    def get_block(self) -> int:
        if not self._online:
            return 0
        try:
            return self._subtensor.block
        except Exception:
            return 0

    def set_weights(
        self,
        netuid: int,
        uids: List[int],
        weights: List[float],
        wallet_name: Optional[str] = None,
    ) -> bool:
        """
        Set consensus weights on-chain.

        In online mode, calls subtensor.set_weights(). In offline mode,
        logs the weights and returns True.
        """
        if not self._online:
            logger.info(
                "[offline] set_weights(netuid=%d, uids=%s, weights=%s)",
                netuid,
                uids[:5],
                [round(w, 4) for w in weights[:5]],
            )
            return True

        try:
            import bittensor as bt

            if wallet_name:
                wallet = bt.wallet(name=wallet_name)
            else:
                wallet = self._wallet

            weight_tensor = np.array(weights, dtype=np.float32)
            total = weight_tensor.sum()
            if total > 0:
                weight_tensor = weight_tensor / total

            uid_tensor = np.array(uids, dtype=np.int64)

            success = self._subtensor.set_weights(
                netuid=netuid,
                wallet=wallet,
                uids=uid_tensor,
                weights=weight_tensor,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

            logger.info(
                "set_weights on netuid=%d: %s (%d uids)",
                netuid,
                "success" if success else "failed",
                len(uids),
            )
            return bool(success)

        except Exception as e:
            logger.warning("set_weights failed: %s", e)
            return False

    def get_metagraph(self, netuid: int) -> Optional[Any]:
        """Retrieve the metagraph for a subnet."""
        if not self._online:
            return None
        try:
            import bittensor as bt
            return bt.metagraph(netuid=netuid, network=self.config.endpoint)
        except Exception as e:
            logger.debug("get_metagraph failed: %s", e)
            return None

    def get_metagraph_summary(self, netuid: int) -> str:
        """Human-readable metagraph summary."""
        mg = self.get_metagraph(netuid)
        if mg is None:
            return "Metagraph unavailable (offline mode)"
        try:
            return (
                f"Metagraph netuid={netuid}: "
                f"n={mg.n}, "
                f"block={mg.block}, "
                f"total_stake={float(mg.total_stake):.2f}"
            )
        except Exception:
            return f"Metagraph netuid={netuid}: connected"


class InsigniaEmulator:
    """
    Full emulator for the Insignia subnet on Bittensor testnet.

    Runs iterative epochs where:
      1. A parameter configuration is decoded
      2. Simulated L1/L2 agents produce submissions
      3. The validator scores submissions using the Insignia scoring engine
      4. Weights are set on-chain (or logged in offline mode)
      5. Attack detection evaluates resilience
      6. Fitness metrics are computed
      7. (Optional) NSGA-II proposes next parameter configuration

    This loop enables rapid iteration on incentive mechanism design
    with real consensus feedback before mainnet deployment.
    """

    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.chain = ChainInterface(config)
        self.detector = AttackDetector()
        self.results: List[EmulatorEpochResult] = []

    def initialize(self) -> bool:
        """
        Initialize the emulator. Attempts chain connection; falls back
        to offline mode gracefully.
        """
        logger.info("=" * 70)
        logger.info("  INSIGNIA TESTNET EMULATOR")
        logger.info("  Network: %s (%s)", self.config.network.value, self.config.endpoint)
        logger.info("=" * 70)

        connected = self.chain.connect()
        mode = "ONLINE" if connected else "OFFLINE"
        logger.info("Mode: %s", mode)

        if connected and self.config.netuid:
            summary = self.chain.get_metagraph_summary(self.config.netuid)
            logger.info("Subnet: %s", summary)

        return True

    def run_single_epoch(
        self,
        param_vector: Optional[np.ndarray] = None,
        epoch_idx: int = 0,
        l1_agents: Optional[List[MinerAgent]] = None,
        l2_agents: Optional[List[TraderAgent]] = None,
    ) -> EmulatorEpochResult:
        """
        Execute a single emulator epoch with the given parameter configuration.

        If no param_vector is provided, uses defaults.
        If no agents are provided, creates the default agent mix.
        """
        t0 = time.time()

        if param_vector is None:
            param_vector = encode_defaults()

        config = decode(param_vector)

        if l1_agents is None or l2_agents is None:
            l1_agents, l2_agents = create_default_agents(
                n_honest=self.config.n_honest_l1,
                n_overfitters=max(1, self.config.n_adversarial_l1 // 3),
                n_copycats=max(1, self.config.n_adversarial_l1 // 4),
                n_gamers=max(1, self.config.n_adversarial_l1 // 4),
                n_sybils=max(1, self.config.n_adversarial_l1 // 3),
                n_random=1,
                n_honest_traders=self.config.n_honest_l2,
                n_copy_traders=self.config.n_adversarial_l2,
            )

        harness = SimulationHarness(
            l1_agents=l1_agents,
            l2_agents=l2_agents,
            n_epochs=self.config.n_l1_epochs,
            n_trading_steps=self.config.n_l2_trading_steps,
        )

        logger.info("Epoch %d: Running simulation (%d L1, %d L2 agents)...",
                     epoch_idx, len(l1_agents), len(l2_agents))

        sim_result = harness.run(param_vector)
        breach_report = self.detector.evaluate(sim_result)
        fitness = compute_fitness(sim_result, breach_report)

        export_simulation_metrics(sim_result, breach_report, generation=epoch_idx)

        on_chain_weights = {}
        if sim_result.miner_scores:
            uids = list(range(len(sim_result.miner_scores)))
            weights = list(sim_result.miner_scores.values())

            total = sum(weights)
            if total > 0:
                norm_weights = [w / total for w in weights]
            else:
                norm_weights = [1.0 / len(weights)] * len(weights)

            if self.config.netuid is not None:
                self.chain.set_weights(
                    netuid=self.config.netuid,
                    uids=uids,
                    weights=norm_weights,
                )

            on_chain_weights = {
                uid: round(w, 6)
                for uid, w in zip(sim_result.miner_scores.keys(), norm_weights)
            }

        metagraph_snap = None
        block = 0
        if self.config.netuid is not None:
            metagraph_snap = self.chain.get_metagraph_summary(self.config.netuid)
            block = self.chain.get_block()

        elapsed = time.time() - t0

        epoch_result = EmulatorEpochResult(
            epoch=epoch_idx,
            param_vector=param_vector,
            config_summary=summarize_config(config),
            sim_result=sim_result,
            breach_report=breach_report,
            fitness=fitness,
            on_chain_weights=on_chain_weights,
            metagraph_snapshot=metagraph_snap,
            chain_block=block,
            elapsed_seconds=elapsed,
        )

        self._log_epoch_result(epoch_result)
        self.results.append(epoch_result)
        return epoch_result

    def run_parameter_sweep(
        self,
        param_vectors: List[np.ndarray],
    ) -> EmulatorRunResult:
        """
        Run the emulator across multiple parameter configurations.

        Each configuration gets a full simulation + scoring + attack detection
        + on-chain weight-setting cycle. Results are aggregated and the best
        configuration is identified.
        """
        logger.info("Starting parameter sweep (%d configurations)...", len(param_vectors))
        t0 = time.time()

        run_result = EmulatorRunResult(config=self.config.to_dict())
        best_scalarized = float("inf")

        for i, pv in enumerate(param_vectors):
            epoch_result = self.run_single_epoch(param_vector=pv, epoch_idx=i)
            run_result.epochs.append(epoch_result)

            if epoch_result.fitness is not None:
                scalarized = (
                    1.0 * epoch_result.fitness[0]
                    + 2.0 * epoch_result.fitness[1]
                    + 0.5 * epoch_result.fitness[2]
                    + 0.5 * epoch_result.fitness[3]
                )
                if scalarized < best_scalarized:
                    best_scalarized = scalarized
                    run_result.best_epoch = i
                    run_result.best_fitness = epoch_result.fitness
                    run_result.best_config = decode(pv)

        run_result.total_elapsed = time.time() - t0
        return run_result

    def run_evolutionary_tuning(
        self,
        n_generations: int = 20,
        population_size: int = 15,
    ) -> EmulatorRunResult:
        """
        Run the full evolutionary optimization loop with testnet integration.

        Each generation:
          1. NSGA-II proposes a population of parameter vectors
          2. Each individual is evaluated via run_single_epoch()
          3. Fitness feeds back into the evolutionary algorithm
          4. Best configurations are tracked

        Falls back to random search if pymoo is not available.
        """
        from tuning.parameter_space import get_bounds, repair_weights

        logger.info(
            "Starting evolutionary tuning: %d generations x %d population",
            n_generations,
            population_size,
        )

        lower, upper = get_bounds()
        t0 = time.time()

        run_result = EmulatorRunResult(config=self.config.to_dict())
        best_scalarized = float("inf")
        all_vectors = []

        defaults = encode_defaults()
        all_vectors.append(defaults)
        for _ in range(population_size - 1):
            x = np.random.uniform(lower, upper)
            x = repair_weights(x)
            all_vectors.append(x)

        for gen in range(n_generations):
            logger.info("=== Generation %d/%d ===", gen + 1, n_generations)

            gen_fitness = []
            for i, pv in enumerate(all_vectors):
                epoch_idx = gen * population_size + i
                epoch_result = self.run_single_epoch(
                    param_vector=pv, epoch_idx=epoch_idx
                )
                run_result.epochs.append(epoch_result)

                if epoch_result.fitness is not None:
                    gen_fitness.append((i, epoch_result.fitness, pv))

                    scalarized = (
                        1.0 * epoch_result.fitness[0]
                        + 2.0 * epoch_result.fitness[1]
                        + 0.5 * epoch_result.fitness[2]
                        + 0.5 * epoch_result.fitness[3]
                    )
                    if scalarized < best_scalarized:
                        best_scalarized = scalarized
                        run_result.best_epoch = epoch_idx
                        run_result.best_fitness = epoch_result.fitness
                        run_result.best_config = decode(pv)

            if gen < n_generations - 1:
                all_vectors = self._evolve_population(
                    gen_fitness, lower, upper, population_size
                )

        run_result.total_elapsed = time.time() - t0
        logger.info(
            "Evolutionary tuning complete: %.1fs, best at epoch %d",
            run_result.total_elapsed,
            run_result.best_epoch,
        )
        return run_result

    def _evolve_population(
        self,
        gen_fitness: List[Tuple[int, np.ndarray, np.ndarray]],
        lower: np.ndarray,
        upper: np.ndarray,
        pop_size: int,
    ) -> List[np.ndarray]:
        """
        Simple evolutionary step: tournament selection + SBX crossover + mutation.
        Used when pymoo is not available for the integrated emulator loop.
        """
        from tuning.parameter_space import repair_weights

        if not gen_fitness:
            return [
                repair_weights(np.random.uniform(lower, upper))
                for _ in range(pop_size)
            ]

        gen_fitness.sort(key=lambda x: (
            1.0 * x[1][0] + 2.0 * x[1][1] + 0.5 * x[1][2] + 0.5 * x[1][3]
        ))

        elite_count = max(2, pop_size // 5)
        elites = [gf[2].copy() for gf in gen_fitness[:elite_count]]

        next_pop = list(elites)
        parents = [gf[2] for gf in gen_fitness[:max(2, len(gen_fitness) // 2)]]

        while len(next_pop) < pop_size:
            p1 = parents[np.random.randint(len(parents))]
            p2 = parents[np.random.randint(len(parents))]

            beta = np.random.uniform(0.0, 1.0, len(p1))
            child = beta * p1 + (1 - beta) * p2

            mutation_mask = np.random.random(len(child)) < 0.1
            mutation = np.random.uniform(lower, upper)
            child[mutation_mask] = mutation[mutation_mask]

            child = np.clip(child, lower, upper)
            child = repair_weights(child)
            next_pop.append(child)

        return next_pop[:pop_size]

    def save_results(self, output_dir: Optional[str] = None) -> Path:
        """Save all emulator results to disk."""
        out_path = Path(output_dir or self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        all_results = {
            "config": self.config.to_dict(),
            "n_epochs": len(self.results),
            "mode": "online" if self.chain.is_online else "offline",
            "epochs": [r.to_dict() for r in self.results],
        }

        results_file = out_path / "emulator_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        if self.results:
            best_idx = 0
            best_scalarized = float("inf")
            for i, r in enumerate(self.results):
                if r.fitness is not None:
                    s = 1.0 * r.fitness[0] + 2.0 * r.fitness[1] + 0.5 * r.fitness[2] + 0.5 * r.fitness[3]
                    if s < best_scalarized:
                        best_scalarized = s
                        best_idx = i

            best_result = self.results[best_idx]
            if best_result.param_vector is not None:
                np.save(str(out_path / "best_params.npy"), best_result.param_vector)

                import yaml
                best_config = decode(best_result.param_vector)
                yaml_config = {
                    "weight_config": {
                        k: round(v, 6) for k, v in best_config["raw_params"].items()
                        if k.startswith("l1_") or k.startswith("l2_")
                    },
                    "overfitting": best_config["overfitting"],
                    "feedback": best_config["feedback"],
                    "anti_gaming": best_config["anti_gaming"],
                    "trading": best_config["trading"],
                    "buyback": best_config["buyback"],
                }
                with open(out_path / "best_config.yaml", "w") as f:
                    yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        logger.info("Results saved to %s", out_path)
        return out_path

    def _log_epoch_result(self, result: EmulatorEpochResult):
        """Log a concise summary of an epoch result."""
        sim = result.sim_result
        breach = result.breach_report

        honest_mean = (
            float(np.mean(sim.honest_l1_scores)) if sim and sim.honest_l1_scores else 0.0
        )
        adv_mean = (
            float(np.mean(sim.adversarial_l1_scores))
            if sim and sim.adversarial_l1_scores
            else 0.0
        )
        n_breached = breach.n_breached if breach else 0
        total_attacks = breach.total_attacks if breach else 0

        logger.info(
            "  Epoch %d: honest=%.4f adv=%.4f breaches=%d/%d time=%.1fs%s",
            result.epoch,
            honest_mean,
            adv_mean,
            n_breached,
            total_attacks,
            result.elapsed_seconds,
            f" block={result.chain_block}" if result.chain_block > 0 else "",
        )
