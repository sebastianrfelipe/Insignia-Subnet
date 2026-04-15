"""
Evolutionary Multi-Objective Optimizer

Uses pymoo's NSGA-II algorithm to search the parameter space for
configurations that maximize honest miner performance while minimizing
attack vector breaches.

Objectives (all minimized):
  1. -mean_honest_score: Negative mean composite score of honest miners
  2. attack_breach_rate: Fraction of attack vectors breached
  3. score_variance: Variance of honest miner scores (prefer stability)
  4. -score_separation: Negative gap between honest and adversarial scores

The optimizer uses a repair operator to enforce weight-sum constraints
and exports metrics to Prometheus after each generation.
"""

from __future__ import annotations

import logging
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination.default import DefaultMultiObjectiveTermination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tuning.parameter_space import (
    N_PARAMS, PARAM_NAMES, PARAMETER_DEFINITIONS,
    get_bounds, get_group_indices, repair_weights, decode, encode_defaults,
    summarize_config,
)
from tuning.simulation import SimulationHarness, create_default_agents, SimulationResult
from tuning.attack_detector import AttackDetector, BreachReport
from tuning.metrics_exporter import export_simulation_metrics, export_optimizer_metrics

logger = logging.getLogger("optimizer")
logger.setLevel(logging.INFO)

N_OBJECTIVES = 4


def compute_fitness(
    sim_result: SimulationResult,
    breach_report: BreachReport,
) -> np.ndarray:
    """
    Compute the multi-objective fitness vector from simulation results.
    All objectives are minimized.

    Returns: [neg_honest_score, breach_rate, score_variance, neg_separation]
    """
    honest = sim_result.honest_l1_scores
    adversarial = sim_result.adversarial_l1_scores

    mean_honest = float(np.mean(honest)) if honest else 0.0
    neg_honest_score = -mean_honest

    breach_rate = breach_report.breach_rate

    score_variance = float(np.var(honest)) if len(honest) > 1 else 0.0

    mean_adversarial = float(np.mean(adversarial)) if adversarial else mean_honest
    separation = mean_honest - mean_adversarial
    neg_separation = -separation

    return np.array([neg_honest_score, breach_rate, score_variance, neg_separation])


OBJECTIVE_NAMES = [
    "neg_honest_score",
    "breach_rate",
    "score_variance",
    "neg_separation",
]


# ---------------------------------------------------------------------------
# pymoo Weight Repair Operator
# ---------------------------------------------------------------------------

if PYMOO_AVAILABLE:
    class WeightRepairOperator(Repair):
        """Repair L1/L2 weights to sum to 1.0 after crossover/mutation."""

        def _do(self, problem, X, **kwargs):
            for i in range(len(X)):
                X[i] = repair_weights(X[i])
            return X


# ---------------------------------------------------------------------------
# pymoo Problem Definition
# ---------------------------------------------------------------------------

if PYMOO_AVAILABLE:
    class InsigniaTuningProblem(Problem):
        """
        Multi-objective optimization problem for Insignia subnet parameters.

        Each evaluation runs a full simulation with all agent types and
        computes the 4-objective fitness vector.
        """

        def __init__(
            self,
        n_honest: int = 6,
            n_adversarial_each: int = 1,
        n_epochs: int = 100,
            n_trading_steps: int = 150,
        ):
            lower, upper = get_bounds()
            super().__init__(
                n_var=N_PARAMS,
                n_obj=N_OBJECTIVES,
                xl=lower,
                xu=upper,
            )
            self.n_honest = n_honest
            self.n_adversarial_each = n_adversarial_each
            self.n_epochs = n_epochs
            self.n_trading_steps = n_trading_steps
            self.detector = AttackDetector()
            self._eval_count = 0
            self._generation = 0

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((len(X), N_OBJECTIVES))

            for i, x in enumerate(X):
                x_repaired = repair_weights(x)
                l1_agents, l2_agents = create_default_agents(
                    n_honest=self.n_honest,
                    n_overfitters=self.n_adversarial_each,
                    n_copycats=self.n_adversarial_each,
                    n_gamers=self.n_adversarial_each,
                    n_sybils=min(2, self.n_adversarial_each * 2),
                    n_random=1,
                    n_honest_traders=max(2, self.n_honest // 3),
                    n_copy_traders=self.n_adversarial_each,
                )

                harness = SimulationHarness(
                    l1_agents=l1_agents,
                    l2_agents=l2_agents,
                    n_epochs=self.n_epochs,
                    n_trading_steps=self.n_trading_steps,
                )

                try:
                    sim_result = harness.run(x_repaired)
                    breach_report = self.detector.evaluate(sim_result)
                    fitness = compute_fitness(sim_result, breach_report)

                    export_simulation_metrics(
                        sim_result, breach_report,
                        generation=self._generation, individual=i,
                    )
                except Exception as e:
                    logger.warning("Evaluation failed for individual %d: %s", i, e)
                    fitness = np.array([0.0, 1.0, 1.0, 0.0])

                F[i] = fitness
                self._eval_count += 1

            out["F"] = F

else:
    InsigniaTuningProblem = None


# ---------------------------------------------------------------------------
# Fallback: Random Search (when pymoo is not available)
# ---------------------------------------------------------------------------

class RandomSearchOptimizer:
    """
    Simple random search fallback when pymoo is not installed.
    Samples random parameter vectors, evaluates them, and keeps
    the best according to a weighted scalarized objective.
    """

    def __init__(
        self,
        n_iterations: int = 50,
        n_honest: int = 6,
        n_epochs: int = 100,
        n_trading_steps: int = 150,
    ):
        self.n_iterations = n_iterations
        self.n_honest = n_honest
        self.n_epochs = n_epochs
        self.n_trading_steps = n_trading_steps
        self.detector = AttackDetector()
        self.results: List[Dict] = []
        self.best_fitness = np.inf
        self.best_params: Optional[np.ndarray] = None
        self.best_config: Optional[Dict] = None

    def run(self) -> Dict[str, Any]:
        lower, upper = get_bounds()
        defaults = encode_defaults()

        # Always evaluate defaults first
        candidates = [defaults]
        for _ in range(self.n_iterations - 1):
            x = np.random.uniform(lower, upper)
            x = repair_weights(x)
            candidates.append(x)

        for i, x in enumerate(candidates):
            logger.info("Random search: evaluating %d/%d", i + 1, len(candidates))

            l1_agents, l2_agents = create_default_agents(
                n_honest=self.n_honest,
                n_overfitters=1, n_copycats=1, n_gamers=1,
                n_sybils=2, n_random=1,
                n_honest_traders=2, n_copy_traders=1,
            )
            harness = SimulationHarness(
                l1_agents=l1_agents, l2_agents=l2_agents,
                n_epochs=self.n_epochs, n_trading_steps=self.n_trading_steps,
            )

            try:
                sim_result = harness.run(x)
                breach_report = self.detector.evaluate(sim_result)
                fitness = compute_fitness(sim_result, breach_report)

                # Scalarize: weighted sum of objectives
                scalarized = (
                    1.0 * fitness[0]     # honest score (neg, so lower is better)
                    + 2.0 * fitness[1]   # breach rate (heavily penalize)
                    + 0.5 * fitness[2]   # variance
                    + 0.5 * fitness[3]   # separation
                )

                entry = {
                    "iteration": i,
                    "fitness": fitness.tolist(),
                    "scalarized": float(scalarized),
                    "breach_rate": float(fitness[1]),
                    "honest_score": float(-fitness[0]),
                }
                self.results.append(entry)

                export_simulation_metrics(sim_result, breach_report, generation=0, individual=i)

                if scalarized < self.best_fitness:
                    self.best_fitness = scalarized
                    self.best_params = x.copy()
                    self.best_config = decode(x)
                    logger.info(
                        "  New best: scalarized=%.4f honest=%.4f breaches=%d/%d",
                        scalarized, -fitness[0],
                        breach_report.n_breached, breach_report.total_attacks,
                    )

            except Exception as e:
                logger.warning("Evaluation %d failed: %s", i, e)

        return {
            "best_fitness": self.best_fitness,
            "best_config": self.best_config,
            "best_params": self.best_params.tolist() if self.best_params is not None else None,
            "all_results": self.results,
        }


# ---------------------------------------------------------------------------
# NSGA-II Runner
# ---------------------------------------------------------------------------

def run_nsga2(
    n_generations: int = 20,
    population_size: int = 30,
    n_honest: int = 6,
    n_adversarial_each: int = 1,
    n_epochs: int = 100,
    n_trading_steps: int = 150,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the NSGA-II multi-objective optimizer.

    Returns a dict with the Pareto front, best configs, and history.
    """
    if not PYMOO_AVAILABLE:
        logger.warning("pymoo not installed — falling back to random search")
        rs = RandomSearchOptimizer(
            n_iterations=population_size,
            n_honest=n_honest,
            n_epochs=n_epochs,
            n_trading_steps=n_trading_steps,
        )
        return rs.run()

    problem = InsigniaTuningProblem(
        n_honest=n_honest,
        n_adversarial_each=n_adversarial_each,
        n_epochs=n_epochs,
        n_trading_steps=n_trading_steps,
    )

    algorithm = NSGA2(
        pop_size=population_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=WeightRepairOperator(),
    )

    termination = DefaultMultiObjectiveTermination(
        n_max_gen=n_generations,
        period=5,
    )

    logger.info("Starting NSGA-II: pop=%d, gen=%d, params=%d, objectives=%d",
                population_size, n_generations, N_PARAMS, N_OBJECTIVES)
    t0 = time.time()

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
    )

    elapsed = time.time() - t0
    logger.info("Optimization complete in %.1fs (%d evaluations)",
                elapsed, problem._eval_count)

    pareto_front = res.F if res.F is not None else np.array([])
    pareto_X = res.X if res.X is not None else np.array([])

    # Select knee point (closest to ideal point)
    best_idx = 0
    if len(pareto_front) > 0:
        ideal = pareto_front.min(axis=0)
        nadir = pareto_front.max(axis=0)
        rng = nadir - ideal
        rng[rng < 1e-12] = 1.0
        normalized = (pareto_front - ideal) / rng
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        best_idx = int(np.argmin(distances))

    best_x = pareto_X[best_idx] if len(pareto_X) > 0 else encode_defaults()
    best_config = decode(best_x)

    # Export final optimizer metrics
    if len(pareto_front) > 0:
        best_f = pareto_front[best_idx]
        export_optimizer_metrics(
            generation=n_generations,
            best_fitness={
                OBJECTIVE_NAMES[j]: float(best_f[j])
                for j in range(N_OBJECTIVES)
            },
            pareto_size=len(pareto_front),
            diversity=float(np.mean(np.std(pareto_X, axis=0))) if len(pareto_X) > 1 else 0.0,
        )

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = {
        "n_generations": n_generations,
        "population_size": population_size,
        "n_evaluations": problem._eval_count,
        "elapsed_seconds": round(elapsed, 1),
        "pareto_front_size": len(pareto_front),
        "best_index": best_idx,
        "best_fitness": pareto_front[best_idx].tolist() if len(pareto_front) > 0 else [],
        "best_config_summary": summarize_config(best_config),
        "objective_names": OBJECTIVE_NAMES,
    }

    with open(out_path / "optimization_result.json", "w") as f:
        json.dump(result, f, indent=2)

    if len(pareto_front) > 0:
        np.save(str(out_path / "pareto_front.npy"), pareto_front)
        np.save(str(out_path / "pareto_X.npy"), pareto_X)

    np.save(str(out_path / "best_params.npy"), best_x)

    logger.info("Results saved to %s", out_path)
    logger.info("Pareto front: %d solutions", len(pareto_front))
    logger.info("Best (knee point):\n%s", summarize_config(best_config))

    return {
        "result": result,
        "best_config": best_config,
        "best_params": best_x,
        "pareto_front": pareto_front,
        "pareto_X": pareto_X,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Insignia Parameter Optimizer")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--n-honest", type=int, default=6)
    parser.add_argument("--n-adversarial", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_nsga2(
        n_generations=args.generations,
        population_size=args.population,
        n_honest=args.n_honest,
        n_adversarial_each=args.n_adversarial,
        n_epochs=args.n_epochs,
        n_trading_steps=args.n_steps,
        output_dir=args.output,
        seed=args.seed,
    )

    print("\n=== Optimization Complete ===")
    if "result" in result:
        r = result["result"]
        print(f"Pareto front: {r['pareto_front_size']} solutions")
        print(f"Evaluations: {r['n_evaluations']}")
        print(f"Time: {r['elapsed_seconds']}s")
        if r["best_fitness"]:
            for name, val in zip(OBJECTIVE_NAMES, r["best_fitness"]):
                print(f"  {name}: {val:.4f}")
