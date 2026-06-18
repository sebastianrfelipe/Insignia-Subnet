#!/usr/bin/env python
"""
Insignia Subnet — Full End-to-End Demo (Paired Genetic Mechanism)

Demonstrates the single incentive mechanism end-to-end:

  1. Researcher miners train ML models; trader miners build strategies.
  2. The validator forms chain-seeded (researcher, trader) pairs.
  3. Each pair is jointly evaluated: the model on the benchmark + the trader's
     strategy running on that model -> a multi-objective fitness vector.
  4. NSGA-II non-dominated sorting + crowding rank the pairs.
  5. A collusion-graph screen + variance-penalized marginal-contribution credit
     produce a single Yuma weight vector over all miners.
  6. Pairs are bred (elite retention + crossover + mutation) for the next
     generation.

Run this to see the full pipeline with synthetic data. No external
dependencies beyond numpy, pandas, sklearn, joblib.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --n-researchers 8 --n-traders 5 --n-generations 3
"""

from __future__ import annotations

import sys
import os
import argparse
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tuning.simulation import SimulationHarness, create_default_agents
from tuning.parameter_space import encode_defaults, decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("demo")


def section(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info("  %s", title)
    logger.info("=" * 70)


def run_full_demo(
    n_researchers: int = 6,
    n_traders: int = 3,
    n_generations: int = 3,
    n_trading_steps: int = 120,
):
    section("INSIGNIA SUBNET — PAIRED GENETIC MECHANISM DEMO")
    logger.info(
        "  Researchers: %d | Traders: %d | Generations: %d | Trading steps: %d",
        n_researchers, n_traders, n_generations, n_trading_steps,
    )

    # A realistic adversarial mix: honest researchers/traders plus an overfitter,
    # a copycat, a single-metric gamer, a sybil, one colluding ring, and a
    # partner-selection gamer.
    researchers, traders = create_default_agents(
        n_honest=n_researchers,
        n_overfitters=1,
        n_copycats=1,
        n_gamers=1,
        n_sybils=1,
        n_random=0,
        n_honest_traders=n_traders,
        n_copy_traders=1,
        n_colluding_rings=1,
        n_partner_gamers=1,
    )

    section("PHASE 1: Miner population")
    logger.info("  Researcher miners: %s", ", ".join(a.uid for a in researchers))
    logger.info("  Trader miners:     %s", ", ".join(a.uid for a in traders))

    harness = SimulationHarness(
        researcher_agents=researchers,
        trader_agents=traders,
        n_epochs=n_generations,
        n_trading_steps=n_trading_steps,
    )

    section("PHASE 2-5: Chain-seeded pairing -> joint eval -> NSGA-II -> credit")
    result = harness.run(encode_defaults())

    section("GENERATION SUMMARIES")
    for gen in result.generation_summaries:
        logger.info(
            "  Gen %d: %d pairs, pareto front=%d, collusion flags=%d",
            gen["generation"], gen["n_pairs"], gen["pareto_front_size"],
            gen["n_collusion_flags"],
        )

    section("TOP PAIRS (final generation)")
    for p in result.pair_fitnesses[:8]:
        flag = " [COLLUSION]" if p["collusion_flagged"] else ""
        logger.info(
            "  %s: pair=%.3f (model=%.3f trading=%.3f) rank=%d%s",
            p["pair"], p["pair_composite"], p["model_composite"],
            p["trading_composite"], p["rank"], flag,
        )

    section("SINGLE EMISSION VECTOR (Yuma weights)")
    weights = result.pairing_weights
    for uid, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        role = "researcher" if uid in result.miner_types else "trader"
        logger.info("  %-22s (%-10s): %.4f", uid, role, w)

    section("ANTI-GAMING OUTCOMES")
    honest_mean = float(np.mean(result.honest_researcher_scores)) if result.honest_researcher_scores else 0.0
    adv_mean = float(np.mean(result.adversarial_researcher_scores)) if result.adversarial_researcher_scores else 0.0
    logger.info("  Mean honest researcher quality:      %.4f", honest_mean)
    logger.info("  Mean adversarial researcher quality: %.4f", adv_mean)
    logger.info("  Collusion pairs flagged: %s", [k for k, _ in result.collusion_flags])
    if result.colluder_credit:
        logger.info("  Colluder quality (should be below honest mean):")
        for uid, c in result.colluder_credit.items():
            logger.info("    %s: %.4f", uid, c)
    logger.info("  Pairing seed source: %s (partner identity unchooseable)", result.pairing_seed_source)

    section("DEMO COMPLETE")
    logger.info("  Single paired mechanism demonstrated end-to-end.")
    logger.info("  In production: proprietary tick data replaces the synthetic benchmark,")
    logger.info("  a real exchange feed replaces simulated prices, and the chain block")
    logger.info("  hash seeds the pairing.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insignia Paired Mechanism Demo")
    parser.add_argument("--n-researchers", type=int, default=6, help="Number of honest researcher miners")
    parser.add_argument("--n-traders", type=int, default=3, help="Number of honest trader miners")
    parser.add_argument("--n-generations", type=int, default=3, help="Number of genetic generations")
    parser.add_argument("--n-steps", type=int, default=120, help="Trading steps per pair")
    args = parser.parse_args()

    run_full_demo(
        n_researchers=args.n_researchers,
        n_traders=args.n_traders,
        n_generations=args.n_generations,
        n_trading_steps=args.n_steps,
    )
