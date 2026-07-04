"""
Regression test for honest/adversarial separation in the simulation harness.

Grounds EMULATOR_SPEC §9 (acceptance gate: empirical separation >= 0.90) in an
actual harness run with the default 14-agent benchmark population. The spec's
§6.6 "surrogate-vs-empirical" narrative describes a GP surrogate that does not
exist in the codebase — the optimizer already routes through `SimulationHarness`
(see `subnet/tuning/optimizer.py:172-213`). This test pins the harness's actual
separation behavior so any future regression (e.g. an adversary penalty
multiplier at `simulation.py` being silently raised) is caught.

If this test fails below 0.90, the fix is to tighten the adversary penalty
multipliers in the scoring loop of `SimulationHarness.run` (search for
`EXP-ADVERSARY-COVERAGE-002`).
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
    def test_harness_separation_meets_gate(self):
        """Empirical separation (harness, default params, §5.3 population) >= 0.90.

        Per EMULATOR_SPEC §9 acceptance gate. Anti-gaming penalty paths for all
        6 adversary types (Copycat, CopyTrader, SybilMiner, OverfittingMiner,
        SingleMetricGamer, PartnerGamer) plus ColludingResearcher are applied
        in `SimulationHarness.run` per EXP-ADVERSARY-COVERAGE-002. See
        `results/adversary_coverage_analysis.md`.
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
            f"Tighten the adversary penalty multipliers in subnet/tuning/simulation.py.",
        )

    def test_no_adversary_outscores_honest_mean(self):
        """Concrete regression marker: no adversary type may score higher
        than the honest mean. Pins the leak documented in
        `results/adversary_coverage_analysis.md` (SybilMiner previously
        scored 0.9163 > honest 0.9151 because sybil_pressure / ensemble
        signals never fed back into `miner_scores`).
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
            f"Each leaked type needs a tighter penalty multiplier in the "
            f"SimulationHarness.run scoring loop.",
        )


    def test_sybil_penalty_is_signal_driven(self):
        """The SybilMiner penalty must be signal-driven, not a flat constant.

        The pre-fix stopgap applied a static 0.0001 multiplier to all sybils
        regardless of behavior. The signal-driven path (EXP-ADVERSARY-COVERAGE-002
        §1) uses the generation's `sybil_pressure` signal to modulate the
        multiplier within the near-zero regime the §9 gate requires. This test
        pins that the sybil score lands in the signal-driven range — above the
        flat floor (`_SYBIL_FLOOR_MULTIPLIER × base`) and below the no-signal
        ceiling (`_SYBIL_GATE_SCALE × base`) — so a future regression to a flat
        constant is caught.
        """
        from tuning.simulation import (
            _SYBIL_FLOOR_MULTIPLIER,
            _SYBIL_GATE_SCALE,
            _SYBIL_DETECTION_SENSITIVITY,
            _SYBIL_CORRELATION_PENALTY,
        )

        sim_result = _run_harness()
        researcher_types = _per_type_breakdown(sim_result)
        sybil_score = researcher_types.get("sybil")
        self.assertIsNotNone(sybil_score, "No sybil agents in the harness output")

        # Base score for an honest researcher is ~0.915; sybils start from a
        # similar composite before the penalty fires.
        base = 0.915
        # Floor-only score: what a flat _SYBIL_FLOOR_MULTIPLIER would produce.
        floor_score = base * _SYBIL_FLOOR_MULTIPLIER
        # No-signal ceiling: the multiplier when sybil_pressure == 0 (the
        # signal-driven path's maximum output before the floor kicks in).
        no_signal_mult = (1.0 - min(0.95, _SYBIL_DETECTION_SENSITIVITY * _SYBIL_CORRELATION_PENALTY)) * _SYBIL_GATE_SCALE
        ceiling_score = base * no_signal_mult

        # The signal-driven score must be strictly above the flat floor (else
        # the signal is not modulating) and at or below the no-signal ceiling
        # (else the floor is too loose).
        self.assertGreater(
            sybil_score,
            floor_score * 1.5,
            f"Sybil score {sybil_score:.6f} is at the flat floor ({floor_score:.6f}); "
            f"the sybil_pressure signal is not modulating the penalty.",
        )
        self.assertLessEqual(
            sybil_score,
            ceiling_score * 1.1,  # 10% tolerance for base-score variance
            f"Sybil score {sybil_score:.6f} exceeds the no-signal ceiling ({ceiling_score:.6f}); "
            f"the _SYBIL_FLOOR_MULTIPLIER may be too loose.",
        )


if __name__ == "__main__":
    unittest.main()
