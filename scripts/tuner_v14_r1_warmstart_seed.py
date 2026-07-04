"""Tuner prep — fold V14-R1 empirical fitness into an NSGA-II warm-start seed (cycle step 4).

Computes the V14-R1-CORRECTED-KP proxy config's empirical fitness vector using
the optimizer's own `compute_fitness()` function (so the seed is consistent
with what NSGA-II evaluates), then emits a warm-start seed file the tuner
agent can inject into the next optimizer run's initial population.

The seed file contains:
  - `params`: the flat parameter vector (encode_defaults, V14-R1 proxy)
  - `fitness`: the 4-objective fitness vector [neg_honest, breach_rate,
    variance, neg_separation] computed by `compute_fitness()`
  - `config_summary`: human-readable config summary
  - `empirical_separation`: from the step-2 validation
  - `sentinel_breach_rate`: adversary-type breach rate from the step-3 matrix
  - `source_reports`: paths to the step-2 / step-3 artifacts this seed folds in

Usage:
    python scripts/tuner_v14_r1_warmstart_seed.py [--epochs N] [--trading-steps N]
                                                   [--output results/]

The V14-R1-CORRECTED-KP config itself lives in the orchestrator's MongoDB
(memory key `v14_r1_corrected_config`); `encode_defaults` is the repo-side
proxy. When the orchestrator dispatches the tuner agent, it should substitute
the MongoDB config vector for `encode_defaults()` here and re-emit the seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "subnet"))

from tuning.parameter_space import encode_defaults, decode, summarize_config, PARAM_NAMES
from tuning.simulation import SimulationHarness, create_default_agents
from tuning.attack_detector import AttackDetector
from tuning.optimizer import compute_fitness, OBJECTIVE_NAMES, N_OBJECTIVES


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5, help="Generations (default 5)")
    p.add_argument("--trading-steps", type=int, default=120, help="Trading steps per generation (default 120)")
    p.add_argument("--output", type=str, default="results", help="Output directory (default results)")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    print("Tuner prep — computing V14-R1 empirical fitness for NSGA-II warm-start seed...")
    t0 = time.time()

    # Run the harness with the V14-R1 proxy config.
    l1, l2 = create_default_agents(
        n_honest=6, n_overfitters=1, n_copycats=1, n_gamers=1,
        n_sybils=2, n_random=1,
        n_honest_traders=3, n_copy_traders=1, n_colluding_rings=1, n_partner_gamers=1,
    )
    harness = SimulationHarness(
        researcher_agents=l1, trader_agents=l2,
        n_epochs=args.epochs, n_trading_steps=args.trading_steps,
    )
    params = encode_defaults()
    sim_result = harness.run(params)
    print(f"  harness completed in {time.time() - t0:.1f}s")

    # Evaluate through AttackDetector (same path the optimizer uses).
    detector = AttackDetector()
    breach_report = detector.evaluate(sim_result)

    # Compute the 4-objective fitness vector the optimizer minimizes.
    fitness = compute_fitness(sim_result, breach_report)
    print(f"  fitness vector: {dict(zip(OBJECTIVE_NAMES, [round(float(f), 4) for f in fitness]))}")

    # Empirical separation (step-2 gate metric).
    honest_mean = float(np.mean(sim_result.honest_researcher_scores)) if sim_result.honest_researcher_scores else 0.0
    adv_mean = float(np.mean(sim_result.adversarial_researcher_scores)) if sim_result.adversarial_researcher_scores else 0.0
    separation = honest_mean - adv_mean

    # Adversary-type breach rate (step-3 sentinel metric). Only count
    # adversary-type vectors, not non-penalty-path breaches (synthetic harness
    # signals / config-tuning gaps).
    adversary_vector_names = {
        "overfitting_exploitation", "model_plagiarism", "single_metric_gaming",
        "sybil_attack", "copy_trading", "miner_validator_collusion",
        "adversarial_dominance", "insufficient_separation",
    }
    adversary_breaches = [
        b for b in breach_report.breaches
        if b.attack_name in adversary_vector_names and b.breached
    ]
    n_adversary_vectors = len(adversary_vector_names)
    adversary_breach_rate = len(adversary_breaches) / n_adversary_vectors if n_adversary_vectors else 0.0

    config = decode(params)

    seed = {
        "seed_type": "nsga2_warm_start",
        "config_id": "V14-R1-CORRECTED-KP-PROXY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_source": "encode_defaults (Phase 5 proxy; V14-R1-CORRECTED-KP lives in orchestrator MongoDB)",
        "parameters": {
            "epochs": args.epochs,
            "trading_steps": args.trading_steps,
        },
        "params": params.tolist(),
        "param_names": PARAM_NAMES,
        "fitness": {
            "vector": fitness.tolist(),
            "objective_names": OBJECTIVE_NAMES,
            "n_objectives": N_OBJECTIVES,
            "decoded": {
                OBJECTIVE_NAMES[i]: round(float(fitness[i]), 6) for i in range(N_OBJECTIVES)
            },
        },
        "empirical_metrics": {
            "honest_mean": round(honest_mean, 6),
            "adversarial_mean": round(adv_mean, 6),
            "separation": round(separation, 6),
            "separation_gate": 0.90,
            "separation_gate_passed": separation >= 0.90,
            "score_variance": round(float(np.var(sim_result.honest_researcher_scores)), 6) if len(sim_result.honest_researcher_scores) > 1 else 0.0,
            "detector_total_breaches": breach_report.n_breached,
            "detector_total_vectors": breach_report.total_attacks,
            "detector_breach_rate": round(breach_report.breach_rate, 6),
            "adversary_vector_count": n_adversary_vectors,
            "adversary_breaches": len(adversary_breaches),
            "adversary_breach_rate": round(adversary_breach_rate, 6),
            "adversary_surface_clear": len(adversary_breaches) == 0,
        },
        "config_summary": summarize_config(config),
        "source_reports": {
            "step2_simulator": "results/v14_r1_empirical_validation_2026-07-04T02-21-09.json",
            "step3_sentinel": "results/sentinel_coverage_matrix_2026-07-04T02-32-01.json",
            "note": "source reports are from the feat/signal-driven-adversary-penalties branch; "
                    "the orchestrator should re-emit this seed with the live V14-R1-CORRECTED-KP "
                    "config from MongoDB (memory key v14_r1_corrected_config).",
        },
        "consumption": {
            "intended_use": "Inject as an elite individual into the NSGA-II initial population "
                            "for the next tuner run, so the search warm-starts from the V14-R1 "
                            "knee point rather than Latin Hypercube sampling from scratch.",
            "cli_flag": "python -m tuning.optimizer --warm-start <seed.json>",
            "implementation_status": "run_nsga2 warm-start injection: see optimizer.py "
                                     "(--warm-start flag added in this cycle step).",
        },
    }

    json_path = out_dir / f"tuner_v14_r1_warmstart_seed_{ts}.json"
    npy_path = out_dir / f"tuner_v14_r1_warmstart_seed_{ts}.npy"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(seed, f, indent=2, default=str)
    np.save(str(npy_path), params)

    print()
    print(f"V14-R1 empirical fitness (NSGA-II objectives, all minimized):")
    for name, val in zip(OBJECTIVE_NAMES, fitness):
        print(f"  {name}: {float(val):.6f}")
    print()
    print(f"Empirical separation: {separation:.4f} (gate >= 0.90: {'PASS' if separation >= 0.90 else 'FAIL'})")
    print(f"Adversary breach rate: {adversary_breach_rate:.4f} ({len(adversary_breaches)}/{n_adversary_vectors} adversary vectors breached)")
    print(f"  adversary_surface_clear: {seed['empirical_metrics']['adversary_surface_clear']}")
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {npy_path}")

    return 0 if seed["empirical_metrics"]["adversary_surface_clear"] and separation >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
