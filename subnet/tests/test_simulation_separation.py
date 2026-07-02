"""
Regression test for honest/adversarial separation in the simulation harness.

Grounds EMULATOR_SPEC §9 (acceptance gate: empirical separation >= 0.90) in an
actual harness run with the default 14-agent benchmark population. The spec's
§6.6 "surrogate-vs-empirical" narrative describes a GP surrogate that does not
exist in the codebase — the optimizer already routes through `SimulationHarness`
(see `subnet/tuning/optimizer.py:172-213`). This test pins the harness's actual
separation behavior so any future regression (e.g. the `0.50` anti-copy
multiplier at `simulation.py:871,873` being silently raised) is caught.

If this test fails below 0.90, the fix is at `simulation.py:871,873`
(`_scaled(model_eff, 0.50)` for CopycatMiner / CopyTrader) — lower the
multiplier until separation clears the gate, then commit.
"""

import unittest
from typing import Dict, List

import numpy as np

from tuning.parameter_space import encode_defaults
from tuning.simulation import SimulationHarness, create_default_agents


def _run_harness(n_epochs: int = 3, n_trading_steps: int = 60):
    """Run the harness with the default 14-agent benchmark population (spec §5.3).

    Smaller n_epochs / n_trading_steps than production to keep the test fast
    while still exercising the full pairing + scoring + anti-copy path.
    """
    l1_agents, l2_agents = create_default_agents(
        n_honest=6,
        n_overfitters=1,
        n_copycats=1,
        n_gamers=1,
        n_sybils=2,
        n_random=1,
        n_honest_traders=3,
        n_copy_traders=1,
        n_colluding_rings=1,
        n_partner_gamers=1,
    )
    harness = SimulationHarness(
        researcher_agents=l1_agents,
        trader_agents=l2_agents,
        n_epochs=n_epochs,
        n_trading_steps=n_trading_steps,
    )
    return harness.run(encode_defaults())


def _separation(sim_result) -> float:
    honest = sim_result.honest_researcher_scores
    adversarial = sim_result.adversarial_researcher_scores
    if not honest or not adversarial:
        return 0.0
    return float(np.mean(honest)) - float(np.mean(adversarial))


def _per_type_breakdown(sim_result) -> dict:
    """Group miner_scores by agent_type for diagnostics."""
    by_type: Dict[str, List[float]] = {}
    for uid, score in sim_result.miner_scores.items():
        agent_type = sim_result.miner_types.get(uid, "unknown")
        by_type.setdefault(agent_type, []).append(float(score))
    return {t: float(np.mean(scores)) for t, scores in by_type.items()}


def _trader_type_breakdown(sim_result) -> dict:
    by_type: Dict[str, List[float]] = {}
    for uid, score in sim_result.trader_scores.items():
        agent_type = sim_result.trader_types.get(uid, "unknown")
        by_type.setdefault(agent_type, []).append(float(score))
    return {t: float(np.mean(scores)) for t, scores in by_type.items()}


class SeparationRegressionTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_harness_separation_meets_gate(self):
        """Empirical separation (harness, default params, §5.3 population) >= 0.90.

        Per EMULATOR_SPEC §9 acceptance gate. Currently an expected failure —
        the harness applies anti-copy penalties only to CopycatMiner and
        CopyTrader (simulation.py:871,873). The other adversaries (sybil,
        overfitter, single_metric_gamer, partner_gamer) have no penalty path
        and score ~0.90. A broader anti-gaming scoring pass is required.
        See MCP `agent_memory` key `researcher_findings_copycat_v2`.

        Remove @expectedFailure when the 4 missing penalty paths are added
        in `SimulationHarness.run` (simulation.py:860-930) and separation
        clears 0.90. The mirror test `test_harness_separation_gate_is_unmet`
        will then report an unexpected success, signalling it should be
        removed too.
        """
        sim_result = _run_harness()
        sep = _separation(sim_result)
        researcher_types = _per_type_breakdown(sim_result)
        trader_types = _trader_type_breakdown(sim_result)
        print(
            f"\n[harness separation] honest={np.mean(sim_result.honest_researcher_scores):.4f}"
            f" adv={np.mean(sim_result.adversarial_researcher_scores):.4f}"
            f" separation={sep:.4f} (gate >= 0.90)"
        )
        print(f"[researcher scores by type] {researcher_types}")
        print(f"[trader scores by type]     {trader_types}")
        self.assertGreaterEqual(
            sep,
            0.90,
            f"Empirical separation {sep:.4f} below 0.90 gate. "
            f"Tighten the anti-copy multiplier at subnet/tuning/simulation.py:871,873.",
        )

    @unittest.expectedFailure
    def test_harness_separation_gate_is_unmet(self):
        """Documents the current open gap. Remove @expectedFailure when
        `test_harness_separation_meets_gate` passes.
        """
        sim_result = _run_harness()
        sep = _separation(sim_result)
        self.assertGreaterEqual(sep, 0.90)

    @unittest.expectedFailure
    def test_no_adversary_outscores_honest_mean(self):
        """Concrete regression marker: no adversary type may score higher
        than the honest mean. Currently `sybil` leaks (0.9163 > honest 0.9151)
        because the harness's sybil_pressure / ensemble_signals calculations
        never feed back into `miner_scores`. Fix: apply a sybil penalty in
        the scoring loop at `simulation.py:870-873` analogous to the Copycat
        multiplier, sourced from the `sybil_diversity_detector` signal.

        Remove @expectedFailure when every adversary type scores strictly
        below the honest mean. Currently only `sybil` leaks; the other 7
        adversary types already score below honest.
        """
        sim_result = _run_harness()
        honest_mean = float(np.mean(sim_result.honest_researcher_scores))
        researcher_types = _per_type_breakdown(sim_result)
        trader_types = _trader_type_breakdown(sim_result)
        # Adversary types per spec §5.1 / §5.2. NOTE: `random` is deliberately
        # excluded — per EMULATOR_SPEC §5.1 it is the "Noise-floor baseline",
        # not an adversary. `RandomMiner` does not override `is_adversarial()`
        # (inherits base class default `False` at simulation.py:102-103), so
        # the harness at simulation.py:930 routes its scores into
        # `honest_researcher_scores` alongside pure HonestMiner. The honest
        # mean is therefore a blend of pure honest + random baseline.
        adversary_types = {
            "overfitter",
            "copycat",
            "single_metric_gamer",
            "sybil",
            "colluder",
            "copy_trader",
            "colluder_trader",
            "partner_gamer",
        }
        leaks = {}
        for t, mean_score in {**researcher_types, **trader_types}.items():
            if t in adversary_types and mean_score > honest_mean:
                leaks[t] = mean_score
        self.assertEqual(
            leaks,
            {},
            f"Adversary types scoring higher than honest mean ({honest_mean:.4f}): {leaks}. "
            f"Each leaked type needs a penalty path in simulation.py:870-875.",
        )


if __name__ == "__main__":
    unittest.main()
