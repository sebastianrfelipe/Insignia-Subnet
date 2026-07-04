"""Empirical V14-R1 separation validation (cycle step 2).

Runs `SimulationHarness` with the V14-R1-CORRECTED-KP proxy config
(`encode_defaults` — the repo's "Phase 5 secure-and-improving profile")
for multiple trials and generations, then emits a structured report
the orchestrator can consume to clear the §9 separation gate and
unblock steps 3–5 (sentinel, tuner, researcher).

Usage:
    python scripts/validate_v14_r1_separation.py [--trials N] [--epochs N]
                                                  [--trading-steps N]
                                                  [--output results/]

The V14-R1-CORRECTED-KP config itself lives in the orchestrator's
MongoDB (memory key `v14_r1_corrected_config`); `encode_defaults` is
the repo-side proxy that mirrors the Phase 5 profile. When the
orchestrator dispatches the simulator agent, it should substitute the
MongoDB config vector for `encode_defaults()` below.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "subnet"))

from tuning.parameter_space import encode_defaults, decode
from tuning.simulation import SimulationHarness, create_default_agents


def _per_type_breakdown(scores: Dict[str, float], types: Dict[str, str]) -> Dict[str, float]:
    by_type: Dict[str, List[float]] = {}
    for uid, score in scores.items():
        t = types.get(uid, "unknown")
        by_type.setdefault(t, []).append(float(score))
    return {t: float(statistics.fmean(v)) for t, v in by_type.items()}


def _separation(result) -> float:
    h = result.honest_researcher_scores
    a = result.adversarial_researcher_scores
    if not h or not a:
        return 0.0
    return float(np.mean(h)) - float(np.mean(a))


def _breach_rate(result) -> float:
    """Fraction of adversary types that outscored the honest mean."""
    honest_mean = float(np.mean(result.honest_researcher_scores))
    researcher = _per_type_breakdown(result.miner_scores, result.miner_types)
    trader = _per_type_breakdown(result.trader_scores, result.trader_types)
    adversary_types = {
        "overfitter", "copycat", "single_metric_gamer", "sybil", "colluder",
        "copy_trader", "colluder_trader", "partner_gamer",
    }
    n_adv = 0
    n_leaked = 0
    for t, mean_score in {**researcher, **trader}.items():
        if t in adversary_types:
            n_adv += 1
            if mean_score > honest_mean:
                n_leaked += 1
    return (n_leaked / n_adv) if n_adv else 0.0


def run_trial(trial_idx: int, n_epochs: int, n_trading_steps: int) -> Dict[str, Any]:
    """Run one harness trial and return a structured result dict."""
    t0 = time.time()
    l1, l2 = create_default_agents(
        n_honest=6, n_overfitters=1, n_copycats=1, n_gamers=1,
        n_sybils=2, n_random=1,
        n_honest_traders=3, n_copy_traders=1, n_colluding_rings=1, n_partner_gamers=1,
    )
    harness = SimulationHarness(
        researcher_agents=l1, trader_agents=l2,
        n_epochs=n_epochs, n_trading_steps=n_trading_steps,
    )
    result = harness.run(encode_defaults())
    elapsed = time.time() - t0

    sep = _separation(result)
    honest_mean = float(np.mean(result.honest_researcher_scores)) if result.honest_researcher_scores else 0.0
    adv_mean = float(np.mean(result.adversarial_researcher_scores)) if result.adversarial_researcher_scores else 0.0
    breach = _breach_rate(result)

    return {
        "trial": trial_idx,
        "elapsed_s": round(elapsed, 2),
        "separation": sep,
        "honest_mean": honest_mean,
        "adversarial_mean": adv_mean,
        "breach_rate": breach,
        "n_honest": len(result.honest_researcher_scores),
        "n_adversarial": len(result.adversarial_researcher_scores),
        "researcher_scores_by_type": _per_type_breakdown(result.miner_scores, result.miner_types),
        "trader_scores_by_type": _per_type_breakdown(result.trader_scores, result.trader_types),
        "trading_pair_counts": dict(result.trading_pair_counts),
        "n_breach_alerts": len(result.breach_alerts),
        "convergence_indexes": list(result.convergence_indexes),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=3, help="Number of independent harness trials (default 3)")
    p.add_argument("--epochs", type=int, default=5, help="Generations per trial (default 5)")
    p.add_argument("--trading-steps", type=int, default=120, help="Trading steps per generation (default 120)")
    p.add_argument("--output", type=str, default="results", help="Output directory (default results)")
    p.add_argument("--gate", type=float, default=0.90, help="§9 separation gate threshold (default 0.90)")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    print(f"V14-R1 empirical separation validation")
    print(f"  trials={args.trials} epochs={args.epochs} trading_steps={args.trading_steps}")
    print(f"  gate: separation >= {args.gate}")
    print()

    trials = []
    for i in range(args.trials):
        print(f"--- Trial {i + 1}/{args.trials} ---")
        t = run_trial(i, args.epochs, args.trading_steps)
        trials.append(t)
        print(f"  separation={t['separation']:.4f} honest={t['honest_mean']:.4f} "
              f"adv={t['adversarial_mean']:.4f} breach_rate={t['breach_rate']:.2f} "
              f"({t['elapsed_s']:.1f}s)")
        print(f"  researcher_by_type={t['researcher_scores_by_type']}")

    separations = [t["separation"] for t in trials]
    breaches = [t["breach_rate"] for t in trials]
    honest_means = [t["honest_mean"] for t in trials]
    adv_means = [t["adversarial_mean"] for t in trials]

    summary = {
        "report_type": "v14_r1_empirical_separation_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_source": "encode_defaults (Phase 5 proxy; V14-R1-CORRECTED-KP lives in orchestrator MongoDB)",
        "parameters": {
            "trials": args.trials, "epochs": args.epochs,
            "trading_steps": args.trading_steps, "gate": args.gate,
        },
        "summary_stats": {
            "separation_mean": float(statistics.fmean(separations)),
            "separation_min": float(min(separations)),
            "separation_max": float(max(separations)),
            "separation_stdev": float(statistics.pstdev(separations)) if len(separations) > 1 else 0.0,
            "honest_mean_avg": float(statistics.fmean(honest_means)),
            "adversarial_mean_avg": float(statistics.fmean(adv_means)),
            "breach_rate_mean": float(statistics.fmean(breaches)),
            "gate_passed": all(s >= args.gate for s in separations),
            "n_trials_passing_gate": sum(1 for s in separations if s >= args.gate),
        },
        "trials": trials,
    }

    json_path = out_dir / f"v14_r1_empirical_validation_{ts}.json"
    md_path = out_dir / f"v14_r1_empirical_validation_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Human-readable markdown report
    s = summary["summary_stats"]
    lines = [
        f"# V14-R1 Empirical Separation Validation",
        f"",
        f"**Generated:** {summary['generated_at']}",
        f"**Config source:** {summary['config_source']}",
        f"**Parameters:** trials={args.trials}, epochs={args.epochs}, trading_steps={args.trading_steps}",
        f"",
        f"## §9 Separation Gate (threshold >= {args.gate})",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| separation_mean | {s['separation_mean']:.4f} |",
        f"| separation_min | {s['separation_min']:.4f} |",
        f"| separation_max | {s['separation_max']:.4f} |",
        f"| separation_stdev | {s['separation_stdev']:.4f} |",
        f"| honest_mean_avg | {s['honest_mean_avg']:.4f} |",
        f"| adversarial_mean_avg | {s['adversarial_mean_avg']:.4f} |",
        f"| breach_rate_mean | {s['breach_rate_mean']:.2f} |",
        f"| **gate_passed** | **{s['gate_passed']}** |",
        f"| n_trials_passing_gate | {s['n_trials_passing_gate']}/{args.trials} |",
        f"",
        f"## Per-Trial Results",
        f"",
        f"| Trial | Separation | Honest | Adversarial | Breach Rate | Elapsed |",
        f"|---|---|---|---|---|---|",
    ]
    for t in trials:
        lines.append(
            f"| {t['trial']} | {t['separation']:.4f} | {t['honest_mean']:.4f} | "
            f"{t['adversarial_mean']:.4f} | {t['breach_rate']:.2f} | {t['elapsed_s']:.1f}s |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if s["gate_passed"]:
        lines.append(f"✅ **GATE PASSED** — all {args.trials} trials clear §9 separation >= {args.gate}.")
        lines.append("The merged anti-gaming fix (PR #34) + signal-driven SybilMiner penalty hold")
        lines.append("empirically. The cycle may proceed to step 3 (sentinel re-evaluation).")
    else:
        lines.append(f"❌ **GATE FAILED** — {s['n_trials_passing_gate']}/{args.trials} trials passed.")
        lines.append("Tighten the adversary penalty multipliers in `subnet/tuning/simulation.py`")
        lines.append("(search for `EXP-ADVERSARY-COVERAGE-002`) before re-dispatching.")
    lines.append("")
    lines.append(f"_JSON report: `{json_path.name}`_")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print(f"Summary: separation_mean={s['separation_mean']:.4f} "
          f"min={s['separation_min']:.4f} max={s['separation_max']:.4f} "
          f"gate_passed={s['gate_passed']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    return 0 if s["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
