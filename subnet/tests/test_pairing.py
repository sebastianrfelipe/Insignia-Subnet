"""
Tests for the single paired genetic incentive mechanism.

Covers the pairing engine in isolation (deterministic chain-seeded assignment,
NSGA-II ranking, variance-penalized marginal credit, collusion detection) and a
lightweight integration check that honest pairs out-earn adversarial ones.
"""

import unittest
from importlib.util import find_spec

import numpy as np

from insignia.pairing import (
    PairGenome,
    PairFitness,
    PairingConfig,
    PairingPopulation,
    ChainSeededPairing,
    NSGA2Matchmaker,
    MarginalContributionCredit,
    CollusionGraphDetector,
    fast_non_dominated_sort,
    dominates,
)


def _mk(r, t, mc, tc):
    return PairFitness(
        genome=PairGenome(r, t),
        objectives=[-mc, -tc, max(0.0, 0.2 * (1 - tc)), -tc],
        model_composite=mc,
        trading_composite=tc,
        pair_composite=0.5 * mc + 0.5 * tc,
    )


class NonDominatedSortTests(unittest.TestCase):
    def test_dominance(self):
        self.assertTrue(dominates([0.0, 0.0], [1.0, 1.0]))
        self.assertFalse(dominates([0.0, 1.0], [1.0, 0.0]))  # mutually non-dominated

    def test_fronts(self):
        objs = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        fronts = fast_non_dominated_sort(objs)
        self.assertIn(0, fronts[0])  # [0,0] dominates everything -> front 0
        self.assertIn(1, fronts[-1])  # [1,1] dominated by all -> last front


class ChainSeededPairingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PairingConfig(partners_per_miner=3, max_pairs=64)
        self.pairing = ChainSeededPairing(self.cfg)
        self.researchers = [f"r{i}" for i in range(5)]
        self.traders = [f"t{i}" for i in range(5)]

    def test_deterministic(self):
        a = self.pairing.assign(self.researchers, self.traders, "block_42")
        b = self.pairing.assign(self.researchers, self.traders, "block_42")
        self.assertEqual([g.key for g in a], [g.key for g in b])

    def test_seed_changes_assignment(self):
        a = self.pairing.assign(self.researchers, self.traders, "block_1")
        b = self.pairing.assign(self.researchers, self.traders, "block_2")
        self.assertNotEqual([g.key for g in a], [g.key for g in b])

    def test_reproduce_tiebreak_is_order_independent(self):
        # All pairs share an identical selection_score, so reproduction must
        # rely on the deterministic genome-key tiebreaker rather than the input
        # ordering. Two validators holding the same fitnesses in different
        # orders must produce the same next generation.
        fits = []
        for r in self.researchers:
            for t in self.traders:
                f = PairFitness(
                    genome=PairGenome(r, t),
                    objectives=[-0.5, -0.5, 0.1, -0.5],
                    pair_composite=0.5,
                )
                f.selection_score = 0.5
                fits.append(f)

        shuffled = list(reversed(fits))
        a = self.pairing.reproduce(fits, self.researchers, self.traders, "block_7")
        b = self.pairing.reproduce(shuffled, self.researchers, self.traders, "block_7")
        self.assertEqual([g.key for g in a], [g.key for g in b])

    def test_partner_floor(self):
        genomes = self.pairing.assign(self.researchers, self.traders, "block_0")
        r_counts = {r: 0 for r in self.researchers}
        t_counts = {t: 0 for t in self.traders}
        for g in genomes:
            r_counts[g.researcher_uid] += 1
            t_counts[g.trader_uid] += 1
        self.assertGreaterEqual(min(r_counts.values()), self.cfg.partners_per_miner)
        self.assertGreaterEqual(min(t_counts.values()), self.cfg.partners_per_miner)


class CreditAndCollusionTests(unittest.TestCase):
    def test_marginal_credit_normalizes_and_penalizes_variance(self):
        # 'steady' has uniform high scores; 'spiky' has one high + lows.
        fits = [
            _mk("steady", "ta", 0.8, 0.8),
            _mk("steady", "tb", 0.8, 0.8),
            _mk("steady", "tc", 0.8, 0.8),
            _mk("spiky", "ta", 0.95, 0.95),
            _mk("spiky", "tb", 0.2, 0.2),
            _mk("spiky", "tc", 0.2, 0.2),
        ]
        NSGA2Matchmaker().rank(fits)
        credit = MarginalContributionCredit(marginal_weight=0.5).compute(fits)
        self.assertAlmostEqual(sum(credit.values()), 1.0, places=6)
        self.assertGreater(credit["steady"], credit["spiky"])

    def test_collusion_detection_flags_only_non_transferable_lift(self):
        fits = [
            _mk("rc", "tc", 0.95, 0.95),  # colluding ring: great only together
            _mk("rc", "t0", 0.15, 0.15),
            _mk("rc", "t1", 0.15, 0.15),
            _mk("r0", "tc", 0.15, 0.15),
            _mk("r1", "tc", 0.15, 0.15),
            _mk("r0", "t0", 0.82, 0.80),  # honest strong: good with several
            _mk("r0", "t1", 0.80, 0.78),
            _mk("r1", "t0", 0.81, 0.79),
        ]
        NSGA2Matchmaker().rank(fits)
        report = CollusionGraphDetector(fixed_pair_correlation_threshold=0.85).detect(fits)
        flagged = {k for k, _ in report.flagged}
        self.assertIn("rc::tc", flagged)
        self.assertFalse(any(k.startswith("r0::") or k.startswith("r1::") for k in flagged))

    def test_collusion_is_unprofitable_after_discount(self):
        fits = [
            _mk("rc", "tc", 0.95, 0.95),
            _mk("rc", "t0", 0.15, 0.15),
            _mk("rc", "t1", 0.15, 0.15),
            _mk("r0", "tc", 0.15, 0.15),
            _mk("r0", "t0", 0.7, 0.7),
            _mk("r0", "t1", 0.7, 0.7),
            _mk("r1", "t0", 0.7, 0.7),
            _mk("r1", "t1", 0.7, 0.7),
        ]
        pop = PairingPopulation(PairingConfig(partners_per_miner=2))
        out = pop.select(fits)
        weights = out["weights"]
        # The colluding researcher must not out-earn the honest researchers.
        self.assertLess(weights.get("rc", 0.0), weights.get("r0", 0.0))
        self.assertGreaterEqual(out["collusion"].n_flagged, 1)


@unittest.skipIf(
    find_spec("pandas") is None or find_spec("sklearn") is None,
    "simulation dependencies unavailable",
)
class PairedSimulationTests(unittest.TestCase):
    def test_honest_pairs_outscore_adversarial(self):
        from tuning.simulation import SimulationHarness, create_default_agents
        from tuning.parameter_space import encode_defaults

        researchers, traders = create_default_agents(
            n_honest=5, n_overfitters=1, n_copycats=1, n_gamers=1, n_sybils=1,
            n_random=0, n_honest_traders=3, n_copy_traders=1,
            n_colluding_rings=1, n_partner_gamers=1,
        )
        harness = SimulationHarness(
            researcher_agents=researchers, trader_agents=traders,
            n_epochs=3, n_trading_steps=60,
        )
        result = harness.run(encode_defaults())

        self.assertGreater(result.n_pairs, 0)
        self.assertTrue(result.honest_researcher_scores)
        self.assertTrue(result.adversarial_researcher_scores)
        # Honest researchers should, on average, out-score adversarial ones.
        self.assertGreater(
            float(np.mean(result.honest_researcher_scores)),
            float(np.mean(result.adversarial_researcher_scores)),
        )
        # Emission weights form a valid single distribution.
        self.assertAlmostEqual(sum(result.pairing_weights.values()), 1.0, places=5)
        # Colluders should not beat the honest researcher mean.
        if result.colluder_credit:
            honest_mean = float(np.mean(result.honest_researcher_scores))
            self.assertLessEqual(max(result.colluder_credit.values()), honest_mean + 1e-6)


if __name__ == "__main__":
    unittest.main()
