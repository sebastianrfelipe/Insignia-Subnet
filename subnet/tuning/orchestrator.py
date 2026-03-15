"""
Tuning Orchestrator

Top-level entry point that ties together the full automated parameter
tuning pipeline:

  1. Initialize parameter space
  2. Start Prometheus metrics server
  3. Run evolutionary optimizer (NSGA-II or random search fallback)
  4. For each generation: simulate, detect attacks, compute fitness, export metrics
  5. Output Pareto-optimal configurations
  6. Generate summary report and export best config as YAML

Usage:
    python -m tuning.orchestrator --generations 50 --population 30 --output results/
    python -m tuning.orchestrator --mode single    # Run single simulation with defaults
    python -m tuning.orchestrator --mode attack     # Run attack detection only
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tuning.parameter_space import (
    encode_defaults, decode, summarize_config, N_PARAMS, PARAM_NAMES,
)
from tuning.simulation import SimulationHarness, create_default_agents
from tuning.attack_detector import AttackDetector
from tuning.metrics_exporter import (
    start_metrics_server, stop_metrics_server,
    export_simulation_metrics, export_optimizer_metrics,
)
from tuning.optimizer import run_nsga2, compute_fitness, OBJECTIVE_NAMES, RandomSearchOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


def run_single_simulation(
    output_dir: str = "results",
    n_honest: int = 6,
    n_epochs: int = 2,
    n_trading_steps: int = 200,
) -> Dict[str, Any]:
    """Run a single simulation with default parameters and full reporting."""
    logger.info("Running single simulation with default parameters...")

    defaults = encode_defaults()
    config = decode(defaults)

    l1_agents, l2_agents = create_default_agents(
        n_honest=n_honest,
        n_overfitters=2, n_copycats=1, n_gamers=1,
        n_sybils=2, n_random=1,
        n_honest_traders=3, n_copy_traders=1,
    )

    harness = SimulationHarness(
        l1_agents=l1_agents, l2_agents=l2_agents,
        n_epochs=n_epochs, n_trading_steps=n_trading_steps,
    )

    t0 = time.time()
    sim_result = harness.run(defaults)
    elapsed = time.time() - t0

    detector = AttackDetector()
    breach_report = detector.evaluate(sim_result)
    fitness = compute_fitness(sim_result, breach_report)

    export_simulation_metrics(sim_result, breach_report)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report = {
        "mode": "single_simulation",
        "elapsed_seconds": round(elapsed, 2),
        "n_l1_miners": len(l1_agents),
        "n_l2_miners": len(l2_agents),
        "n_epochs": n_epochs,
        "n_trading_steps": n_trading_steps,
        "l1_scores": {
            uid: round(s, 6) for uid, s in sim_result.miner_scores.items()
        },
        "l1_types": sim_result.miner_types,
        "honest_l1_mean": round(float(np.mean(sim_result.honest_l1_scores)), 6) if sim_result.honest_l1_scores else 0,
        "adversarial_l1_mean": round(float(np.mean(sim_result.adversarial_l1_scores)), 6) if sim_result.adversarial_l1_scores else 0,
        "l2_scores": {
            uid: round(s, 6) for uid, s in sim_result.l2_scores.items()
        },
        "l2_types": sim_result.l2_types,
        "fitness": {
            name: round(float(val), 6)
            for name, val in zip(OBJECTIVE_NAMES, fitness)
        },
        "breach_report": breach_report.to_dict(),
        "config_summary": summarize_config(config),
    }

    with open(out_path / "single_simulation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n%s", "=" * 70)
    logger.info("  SINGLE SIMULATION REPORT")
    logger.info("=" * 70)
    logger.info("  Time: %.1fs", elapsed)
    logger.info("  L1 Honest mean: %.4f", report["honest_l1_mean"])
    logger.info("  L1 Adversarial mean: %.4f", report["adversarial_l1_mean"])
    logger.info("")
    logger.info("  Fitness:")
    for name, val in zip(OBJECTIVE_NAMES, fitness):
        logger.info("    %s: %.4f", name, val)
    logger.info("")
    logger.info("  %s", breach_report.summary())
    logger.info("")
    logger.info("  Report saved to %s", out_path / "single_simulation_report.json")

    return report


def run_attack_analysis(
    output_dir: str = "results",
    n_trials: int = 5,
    n_honest: int = 4,
) -> Dict[str, Any]:
    """Run multiple attack detection trials for statistical robustness."""
    logger.info("Running attack analysis (%d trials)...", n_trials)

    defaults = encode_defaults()
    detector = AttackDetector()
    all_breaches: Dict[str, list] = {}

    for trial in range(n_trials):
        logger.info("  Trial %d/%d", trial + 1, n_trials)

        l1_agents, l2_agents = create_default_agents(
            n_honest=n_honest,
            n_overfitters=1, n_copycats=1, n_gamers=1,
            n_sybils=2, n_random=1,
            n_honest_traders=2, n_copy_traders=1,
        )

        harness = SimulationHarness(
            l1_agents=l1_agents, l2_agents=l2_agents,
            n_epochs=2, n_trading_steps=150,
        )

        sim_result = harness.run(defaults)
        breach_report = detector.evaluate(sim_result)

        for b in breach_report.breaches:
            if b.attack_name not in all_breaches:
                all_breaches[b.attack_name] = []
            all_breaches[b.attack_name].append({
                "breached": b.breached,
                "severity": b.severity,
            })

    report = {"n_trials": n_trials, "attacks": {}}
    for attack_name, trials in all_breaches.items():
        breach_count = sum(1 for t in trials if t["breached"])
        severities = [t["severity"] for t in trials]
        report["attacks"][attack_name] = {
            "breach_rate": breach_count / n_trials,
            "mean_severity": round(float(np.mean(severities)), 4),
            "max_severity": round(float(max(severities)), 4),
            "min_severity": round(float(min(severities)), 4),
        }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "attack_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n%s", "=" * 70)
    logger.info("  ATTACK ANALYSIS REPORT (%d trials)", n_trials)
    logger.info("=" * 70)
    for attack, stats in report["attacks"].items():
        status = "VULNERABLE" if stats["breach_rate"] > 0.5 else "ROBUST" if stats["breach_rate"] == 0 else "PARTIAL"
        logger.info(
            "  [%s] %s: breach_rate=%.0f%% severity=%.2f",
            status, attack, stats["breach_rate"] * 100, stats["mean_severity"],
        )

    return report


def run_optimization(
    output_dir: str = "results",
    n_generations: int = 50,
    population_size: int = 30,
    n_honest: int = 4,
    n_adversarial: int = 1,
    n_epochs: int = 2,
    n_trading_steps: int = 150,
    seed: int = 42,
    metrics_port: int = 8000,
) -> Dict[str, Any]:
    """Run the full evolutionary optimization pipeline."""
    logger.info("Starting optimization pipeline...")
    logger.info("  Generations: %d, Population: %d", n_generations, population_size)
    logger.info("  Parameters: %d, Objectives: %d", N_PARAMS, len(OBJECTIVE_NAMES))

    result = run_nsga2(
        n_generations=n_generations,
        population_size=population_size,
        n_honest=n_honest,
        n_adversarial_each=n_adversarial,
        n_epochs=n_epochs,
        n_trading_steps=n_trading_steps,
        output_dir=output_dir,
        seed=seed,
    )

    if "best_config" in result and result["best_config"]:
        config = result["best_config"]
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        yaml_config = {
            "weight_config": {
                k: round(v, 6) for k, v in config["raw_params"].items()
                if k.startswith("l1_") or k.startswith("l2_")
            },
            "overfitting": config["overfitting"],
            "promotion": {
                "top_n": int(config["raw_params"]["promotion_top_n"]),
                "min_consecutive_epochs": int(config["raw_params"]["promotion_min_consecutive_epochs"]),
                "max_overfitting_score": round(config["raw_params"]["promotion_max_overfitting_score"], 4),
                "max_score_decay_pct": round(config["raw_params"]["promotion_max_score_decay_pct"], 4),
                "expiry_epochs": int(config["raw_params"]["promotion_expiry_epochs"]),
            },
            "feedback": config["feedback"],
            "anti_gaming": config["anti_gaming"],
            "trading": config["trading"],
            "buyback": config["buyback"],
        }

        with open(out_path / "best_config.yaml", "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

        logger.info("Best config exported to %s", out_path / "best_config.yaml")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Insignia Subnet Parameter Tuning Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  single       Run one simulation with default params (quick sanity check)
  attack       Run attack detection analysis across multiple trials
  optimize     Run full evolutionary optimization (NSGA-II)
  random       Run random search optimization (no pymoo required)

Examples:
  python -m tuning.orchestrator --mode single
  python -m tuning.orchestrator --mode attack --trials 10
  python -m tuning.orchestrator --mode optimize --generations 50 --population 30
        """,
    )
    parser.add_argument("--mode", choices=["single", "attack", "optimize", "random"],
                        default="single")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--n-honest", type=int, default=4)
    parser.add_argument("--n-adversarial", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-port", type=int, default=8000)
    parser.add_argument("--no-metrics", action="store_true",
                        help="Disable Prometheus metrics server")
    args = parser.parse_args()

    if not args.no_metrics:
        try:
            start_metrics_server(args.metrics_port)
            logger.info("Prometheus metrics at http://localhost:%d/metrics", args.metrics_port)
        except Exception as e:
            logger.warning("Could not start metrics server: %s", e)

    try:
        if args.mode == "single":
            run_single_simulation(
                output_dir=args.output,
                n_honest=args.n_honest,
                n_epochs=args.n_epochs,
                n_trading_steps=args.n_steps,
            )

        elif args.mode == "attack":
            run_attack_analysis(
                output_dir=args.output,
                n_trials=args.trials,
                n_honest=args.n_honest,
            )

        elif args.mode == "optimize":
            run_optimization(
                output_dir=args.output,
                n_generations=args.generations,
                population_size=args.population,
                n_honest=args.n_honest,
                n_adversarial=args.n_adversarial,
                n_epochs=args.n_epochs,
                n_trading_steps=args.n_steps,
                seed=args.seed,
                metrics_port=args.metrics_port,
            )

        elif args.mode == "random":
            logger.info("Running random search optimization...")
            rs = RandomSearchOptimizer(
                n_iterations=args.population,
                n_honest=args.n_honest,
                n_epochs=args.n_epochs,
                n_trading_steps=args.n_steps,
            )
            result = rs.run()

            out_path = Path(args.output)
            out_path.mkdir(parents=True, exist_ok=True)
            with open(out_path / "random_search_result.json", "w") as f:
                json.dump({
                    "best_fitness": result["best_fitness"],
                    "best_params": result["best_params"],
                    "n_iterations": len(result["all_results"]),
                }, f, indent=2)

            if result["best_config"]:
                logger.info("\nBest configuration found:")
                logger.info(summarize_config(result["best_config"]))

    finally:
        if not args.no_metrics:
            stop_metrics_server()


if __name__ == "__main__":
    main()
